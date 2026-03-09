"""Demo: pick-and-place the cube using FrankaSimController."""
from __future__ import annotations

import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fr3_controller import FrankaSimController


def main() -> None:
    with FrankaSimController(viewer=True, realtime=True) as robot:
        print(f"Home EE pos: {robot.ee_position}")
        print(f"Home joints: {robot.joint_positions}")

        # 1. Open gripper
        print("[1] Opening gripper...")
        robot.gripper_open()

        # 2. Move above the cube (pre-grasp)
        print("[2] Moving to pre-grasp pose...")
        pre_grasp = [0.0, 0.2, 0.0, -2.0, 0.0, 2.2, -0.785]
        robot.move_to_joint_position(pre_grasp, duration=2.0)
        print(f"    EE pos: {robot.ee_position}")

        # 3. Lower to grasp height
        print("[3] Lowering to grasp...")
        grasp_pose = [0.0, 0.45, 0.0, -1.75, 0.0, 2.2, -0.785]
        robot.move_to_joint_position(grasp_pose, duration=1.5)
        print(f"    EE pos: {robot.ee_position}")

        # 4. Close gripper
        print("[4] Closing gripper...")
        robot.gripper_close(duration=0.8)
        print(f"    Gripper width: {robot.gripper_width:.4f} m")

        # 5. Lift up
        print("[5] Lifting...")
        lift_pose = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, -0.785]
        robot.move_to_joint_position(lift_pose, duration=2.0)
        print(f"    EE pos: {robot.ee_position}")

        # 6. Capture D435i image
        print("[6] Capturing wrist camera image...")
        images = robot.get_camera_images()
        print(f"    RGB: {images['rgb'].shape}, Depth range: [{images['depth'].min():.3f}, {images['depth'].max():.3f}] m")

        # 7. Move to place position
        print("[7] Moving to place position...")
        place_pose = [0.8, 0.0, 0.0, -1.57, 0.0, 1.57, -0.785]
        robot.move_to_joint_position(place_pose, duration=2.0)

        # 8. Open gripper (release)
        print("[8] Releasing...")
        robot.gripper_open(duration=0.5)

        # 9. Return home
        print("[9] Returning home...")
        robot.move_to_joint_position([0, 0, 0, -1.57079, 0, 1.57079, -0.7853], duration=2.0)

        print("Done! Press Ctrl+C or close the viewer window.")
        # Keep viewer alive
        while robot._viewer and robot._viewer.is_running():
            robot._step()


if __name__ == "__main__":
    main()
