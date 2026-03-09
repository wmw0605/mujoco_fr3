"""Franka FR3 simulator controller — franky-style API over MuJoCo."""
from __future__ import annotations

import pathlib
import time
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "fr3_table_grasp.xml"

# Joint names in order
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["left_finger_slide", "right_finger_slide"]
ARM_CTRLS = [f"fr3_joint{i}_ctrl" for i in range(1, 8)]
FINGER_CTRLS = ["left_finger_ctrl", "right_finger_ctrl"]

FINGER_OPEN = 0.04   # meters (max opening per finger)
FINGER_CLOSE = 0.0


class FrankaSimController:
    """MuJoCo-based Franka FR3 controller with a franky-like interface.

    Usage::

        robot = FrankaSimController()
        robot.move_to_joint_position([0, 0, 0, -1.57, 0, 1.57, -0.785])
        robot.gripper_close()
        robot.move_to_joint_position([0, 0, 0, -1.57, 0, 1.57, -0.785])
        robot.gripper_open()
    """

    def __init__(
        self,
        xml_path: str | pathlib.Path = DEFAULT_XML,
        viewer: bool = True,
        realtime: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._realtime = realtime
        self._viewer = None

        # Cache actuator indices
        self._arm_ctrl_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ARM_CTRLS
        ]
        self._finger_ctrl_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in FINGER_CTRLS
        ]
        self._arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ARM_JOINTS
        ]
        self._arm_qpos_adrs = [self.model.jnt_qposadr[j] for j in self._arm_joint_ids]

        # Reset to home
        self.reset()

        if viewer:
            self._launch_viewer()

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset to the 'home' keyframe."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ── properties ───────────────────────────────────────────────────────

    @property
    def joint_positions(self) -> np.ndarray:
        """Current 7-DOF arm joint positions (rad)."""
        return np.array([self.data.qpos[a] for a in self._arm_qpos_adrs])

    @property
    def joint_velocities(self) -> np.ndarray:
        """Current 7-DOF arm joint velocities (rad/s)."""
        return np.array([self.data.qvel[a] for a in self._arm_qpos_adrs])

    @property
    def ee_position(self) -> np.ndarray:
        """End-effector (pinch_site) Cartesian position [x, y, z]."""
        return self.data.site("pinch_site").xpos.copy()

    @property
    def ee_orientation(self) -> np.ndarray:
        """End-effector orientation as a 3x3 rotation matrix."""
        return self.data.site("pinch_site").xmat.reshape(3, 3).copy()

    @property
    def gripper_width(self) -> float:
        """Total gripper opening width in meters."""
        left = self.data.joint("left_finger_slide").qpos[0]
        right = self.data.joint("right_finger_slide").qpos[0]
        return float(left + right)

    @property
    def sim_time(self) -> float:
        return self.data.time

    # ── motion commands ──────────────────────────────────────────────────

    def move_to_joint_position(
        self,
        target: Sequence[float],
        duration: float = 2.0,
    ) -> None:
        """Move arm to target joint positions over *duration* seconds.

        Uses linear interpolation from current position to target.
        """
        target = np.asarray(target, dtype=np.float64)
        assert target.shape == (7,), f"Expected 7 joint targets, got {target.shape}"

        start = self.joint_positions.copy()
        dt = self.model.opt.timestep
        n_steps = max(int(duration / dt), 1)

        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            q_cmd = start + alpha * (target - start)
            for idx, ctrl_id in enumerate(self._arm_ctrl_ids):
                self.data.ctrl[ctrl_id] = q_cmd[idx]
            self._step()

    def move_to_ee_position(
        self,
        target_pos: Sequence[float],
        duration: float = 2.0,
        damping: float = 0.01,
        target_quat: Sequence[float] | None = None,
    ) -> None:
        """Move end-effector to *target_pos* via damped-least-squares IK.

        If *target_quat* (w, x, y, z) is given, also controls orientation.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        dt = self.model.opt.timestep
        n_steps = max(int(duration / dt), 1)

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        arm_dof_ids = [self.model.jnt_dofadr[j] for j in self._arm_joint_ids]

        use_ori = target_quat is not None
        if use_ori:
            target_quat = np.asarray(target_quat, dtype=np.float64)
            target_quat = target_quat / np.linalg.norm(target_quat)

        for _ in range(n_steps):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.site(site_id).xpos
            if not use_ori and np.linalg.norm(pos_err) < 1e-4:
                break

            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
            J_pos = jac_pos[:, arm_dof_ids]

            if use_ori:
                # Orientation error as angle-axis
                site_xmat = self.data.site(site_id).xmat.reshape(3, 3)
                site_quat = np.zeros(4)
                mujoco.mju_mat2Quat(site_quat, site_xmat.flatten())
                # Compute quaternion error
                err_quat = np.zeros(4)
                mujoco.mju_negQuat(err_quat, site_quat)
                mujoco.mju_mulQuat(err_quat, target_quat, err_quat)
                if err_quat[0] < 0:
                    err_quat = -err_quat
                ori_err = 2.0 * err_quat[1:4]  # small-angle approximation

                J_rot = jac_rot[:, arm_dof_ids]
                J_full = np.vstack([J_pos, 0.3 * J_rot])
                err = np.concatenate([pos_err, 0.3 * ori_err])
                lam = damping * np.eye(6)
                dq = J_full.T @ np.linalg.solve(J_full @ J_full.T + lam, err)
            else:
                lam = damping * np.eye(3)
                dq = J_pos.T @ np.linalg.solve(J_pos @ J_pos.T + lam, pos_err)

            if not use_ori and np.linalg.norm(pos_err) < 1e-4:
                break

            max_dq = 0.05
            dq_clamped = np.clip(dq, -max_dq, max_dq)
            q_cmd = self.joint_positions + dq_clamped
            for idx, ctrl_id in enumerate(self._arm_ctrl_ids):
                self.data.ctrl[ctrl_id] = q_cmd[idx]
            self._step()

    # ── gripper ──────────────────────────────────────────────────────────

    def gripper_open(self, width: float = FINGER_OPEN, duration: float = 0.5) -> None:
        """Open gripper to *width* (per-finger distance, default fully open)."""
        self._move_gripper(width, duration)

    def gripper_close(self, width: float = FINGER_CLOSE, duration: float = 0.5) -> None:
        """Close gripper to *width* (0 = fully closed)."""
        self._move_gripper(width, duration)

    def gripper_grasp(self, width: float = 0.0, force: float = 40.0, duration: float = 0.5) -> None:
        """Close gripper with specified grasping force."""
        self._move_gripper(width, duration)

    def _move_gripper(self, width: float, duration: float) -> None:
        dt = self.model.opt.timestep
        n_steps = max(int(duration / dt), 1)
        for _ in range(n_steps):
            for ctrl_id in self._finger_ctrl_ids:
                self.data.ctrl[ctrl_id] = width
            self._step()

    # ── camera ───────────────────────────────────────────────────────────

    def get_camera_images(
        self,
        width: int = 640,
        height: int = 480,
    ) -> dict[str, np.ndarray]:
        """Render RGB and depth from the D435i wrist camera.

        Returns dict with keys 'rgb' (H,W,3 uint8) and 'depth' (H,W float32 meters).
        """
        renderer = mujoco.Renderer(self.model, height, width)

        renderer.update_scene(self.data, camera="d435i_rgb")
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera="d435i_depth")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        depth = np.clip(depth, 0.105, 10.0)

        renderer.close()
        return {"rgb": rgb, "depth": depth}

    # ── internals ────────────────────────────────────────────────────────

    def _step(self) -> None:
        t0 = time.time()
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()
        if self._realtime:
            elapsed = time.time() - t0
            remaining = self.model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _launch_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False, show_right_ui=False,
        )
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._viewer.cam.lookat[:] = [0.1, 0.0, 0.85]
        self._viewer.cam.distance = 2.0
        self._viewer.cam.azimuth = 130
        self._viewer.cam.elevation = -25

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
