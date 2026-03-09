"""GraspNet-based grasping in MuJoCo simulation.

Mirrors the real-world pipeline in franka-graspnet/scripts/franky_graspnet_rs_demo.py:
  RealSense D435i → GraspNet → franky
but replaces hardware with MuJoCo simulation:
  MuJoCo D435i renderer → GraspNet → FrankaSimController
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib
import time

import numpy as np
import mujoco

# ── paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GRASPNET_DIR = PROJECT_DIR / "franka-graspnet"

# Add franka-graspnet subfolders to path
for sub in ["", "franka_graspnet", "models", "dataset", "utils"]:
    p = str(GRASPNET_DIR / sub) if sub else str(GRASPNET_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
# pointnet2 internal imports need the pointnet2 dir itself on sys.path
_pn2 = str(GRASPNET_DIR / "pointnet2")
if _pn2 not in sys.path:
    sys.path.insert(0, _pn2)

# Local imports
sys.path.insert(0, str(SCRIPT_DIR))
from fr3_controller import FrankaSimController


# ── MuJoCo camera helpers ────────────────────────────────────────────────

def get_cam_intrinsics(model: mujoco.MjModel, cam_name: str,
                       width: int, height: int):
    """Compute pinhole intrinsics (fx, fy, cx, cy) from MuJoCo camera fovy."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
    fy = height / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def render_rgbd(model, data, width=640, height=480):
    """Render RGB + depth from the simulated D435i cameras.

    Returns:
        rgb: (H, W, 3) float32 in [0,1]
        depth: (H, W) float32 in meters (OpenCV convention: z forward)
    """
    renderer = mujoco.Renderer(model, height, width)

    # RGB
    renderer.update_scene(data, camera="d435i_rgb")
    rgb = renderer.render().copy().astype(np.float32) / 255.0

    # Depth
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="d435i_depth")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.close()

    # Clamp invalid depth
    depth[depth <= 0] = 0
    depth[depth > 3.0] = 0  # max 3m for tabletop scene

    return rgb, depth


def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Convert depth image to organized point cloud (H, W, 3) — OpenCV convention."""
    h, w = depth.shape
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def get_cam_pose_in_world(model, data, cam_name):
    """Get camera pose (R, t) in world frame.

    Returns R_cam2world (3x3), t_cam2world (3,)
    MuJoCo camera: -Z forward, +X right, +Y up (OpenGL)
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_body = model.cam_bodyid[cam_id]

    # Camera position in world
    t_cam_world = data.cam_xpos[cam_id].copy()
    # Camera rotation matrix (columns = camera axes in world frame, OpenGL convention)
    R_cam_world = data.cam_xmat[cam_id].reshape(3, 3).copy()

    return R_cam_world, t_cam_world


def grasp_cam_to_world(grasp_R_cam, grasp_t_cam, model, data, cam_name="d435i_depth"):
    """Transform grasp pose from camera frame (OpenCV) to world frame.

    GraspNet outputs are in OpenCV camera convention (Z forward, X right, Y down).
    MuJoCo camera is OpenGL (X right, Y up, -Z forward).
    """
    # OpenCV-to-OpenGL transform
    T_cv2gl = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float64)

    # Camera pose in world (OpenGL convention)
    R_gl_world, t_gl_world = get_cam_pose_in_world(model, data, cam_name)

    # Convert grasp from OpenCV camera to OpenGL camera
    grasp_t_gl = T_cv2gl @ grasp_t_cam
    grasp_R_gl = T_cv2gl @ grasp_R_cam

    # OpenGL camera to world
    grasp_t_world = R_gl_world @ grasp_t_gl + t_gl_world
    grasp_R_world = R_gl_world @ grasp_R_gl

    return grasp_R_world, grasp_t_world


def detect_objects_from_depth(depth, model, data, cam_name="d435i_depth",
                              width=640, height=480, table_z=0.76,
                              height_thresh=0.01, exclude_regions=None):
    """Detect objects above the table using depth point cloud.

    Args:
        exclude_regions: list of (x, y, radius) to exclude (e.g. the box).

    Returns list of object centroids in world frame, sorted by distance
    to the robot base (closest first).
    """
    fx, fy, cx, cy = get_cam_intrinsics(model, cam_name, width, height)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    T_cv2gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R_gl_w = data.cam_xmat[cam_id].reshape(3, 3).copy()
    t_gl_w = data.cam_xpos[cam_id].copy()

    # Deproject valid depth pixels to world
    vs, us = np.where(depth > 0.01)
    zs = depth[vs, us]
    pts_cv = np.stack([(us - cx) * zs / fx, (vs - cy) * zs / fy, zs], axis=1)
    pts_world = (R_gl_w @ (T_cv2gl @ pts_cv.T)).T + t_gl_w

    # Filter to table workspace
    mask = (
        (pts_world[:, 0] > -0.1) & (pts_world[:, 0] < 0.7) &
        (pts_world[:, 1] > -0.4) & (pts_world[:, 1] < 0.4) &
        (pts_world[:, 2] > table_z - 0.02) & (pts_world[:, 2] < table_z + 0.20)
    )
    table_pts = pts_world[mask]

    # Object points: above table surface
    obj_pts = table_pts[table_pts[:, 2] > table_z + height_thresh]
    if len(obj_pts) == 0:
        return []

    # Pre-filter: remove points inside exclude regions BEFORE clustering
    if exclude_regions:
        keep = np.ones(len(obj_pts), dtype=bool)
        for ex, ey, er in exclude_regions:
            dist_sq = (obj_pts[:, 0] - ex)**2 + (obj_pts[:, 1] - ey)**2
            keep &= dist_sq >= er**2
        obj_pts = obj_pts[keep]
        if len(obj_pts) == 0:
            return []

    # Simple clustering via connected components on a 2D grid
    from scipy import ndimage
    grid_res = 0.005  # 5mm grid
    xy_min = obj_pts[:, :2].min(axis=0)
    xy_max = obj_pts[:, :2].max(axis=0)
    grid_shape = np.ceil((xy_max - xy_min) / grid_res).astype(int) + 1
    grid = np.zeros(grid_shape, dtype=bool)
    idx = ((obj_pts[:, :2] - xy_min) / grid_res).astype(int)
    idx = np.clip(idx, 0, grid_shape - 1)
    grid[idx[:, 0], idx[:, 1]] = True
    labeled, n_clusters = ndimage.label(grid)

    centroids = []
    for ci in range(1, n_clusters + 1):
        cells = np.argwhere(labeled == ci)
        cell_centers = cells * grid_res + xy_min
        # Find all object points in this cluster's XY bounding box
        cmin = cell_centers.min(axis=0) - grid_res
        cmax = cell_centers.max(axis=0) + grid_res
        in_cluster = (
            (obj_pts[:, 0] >= cmin[0]) & (obj_pts[:, 0] <= cmax[0]) &
            (obj_pts[:, 1] >= cmin[1]) & (obj_pts[:, 1] <= cmax[1])
        )
        cluster_pts = obj_pts[in_cluster]
        if len(cluster_pts) < 10:  # skip noise
            continue
        centroid = cluster_pts.mean(axis=0)
        centroids.append(centroid)

    # Sort by distance to robot base (origin)
    centroids.sort(key=lambda c: c[0]**2 + c[1]**2)
    # Filter out centroids too close to arm base (may be arm geometry)
    arm_base = np.array([-0.18, 0.0])
    centroids = [c for c in centroids
                 if np.linalg.norm(c[:2] - arm_base) > 0.40 and c[0] > 0.20]
    return centroids


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GraspNet grasping in MuJoCo simulation")
    parser.add_argument("--checkpoint_path", type=str,
                        default=str(GRASPNET_DIR / "checkpoints/graspnet/checkpoint-rs.tar"))
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--angle_threshold_deg", type=float, default=60)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--vis-grasps", action="store_true",
                        help="Visualize predicted grasps with Open3D")
    cfgs = parser.parse_args()

    # === 1. Init Robot (MuJoCo) ===
    print("[1] Initializing MuJoCo simulation...")
    robot = FrankaSimController(viewer=not cfgs.no_viewer, realtime=True)

    # Object and box constants
    GRASPABLE_BODIES = ["target_cube", "orange", "lemon", "tomato", "small_box"]
    TABLE_Z = 0.76
    BOX_POS = np.array([0.38, -0.22, TABLE_Z])  # box_container position
    BOX_WALL_TOP_Z = BOX_POS[2] + 0.039 + 0.033  # wall center z + half-height
    BOX_EXCLUDE_RADIUS = 0.12  # radius around box center to exclude from detection

    # === 2. Init GraspNet ===
    print("[2] Loading GraspNet model...")
    from graspnet_infer import GraspNetInfer
    graspnet_infer = GraspNetInfer(cfgs)

    # Camera intrinsics (from MuJoCo model)
    fx, fy, cx, cy = get_cam_intrinsics(robot.model, "d435i_depth",
                                        cfgs.width, cfgs.height)
    print(f"    Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # === 3. Move to home and open gripper ===
    print("[3] Moving to home pose...")
    HOME = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    robot.move_to_joint_position(HOME, duration=1.0)
    robot.gripper_open(duration=0.3)

    print("\n=== Ready. Press Enter to start grasp cycle (Ctrl+C to quit) ===")

    try:
        while True:
            input("Press Enter to capture and grasp...")

            # Check how many objects remain on the table
            objects_on_table = []
            for bname in GRASPABLE_BODIES:
                bz = robot.data.body(bname).xpos[2]
                bxy = robot.data.body(bname).xpos[:2]
                # On table: z near table height, and NOT already inside the box
                on_table = (bz > TABLE_Z - 0.02 and bz < TABLE_Z + 0.10)
                in_box = (abs(bxy[0] - BOX_POS[0]) < 0.07 and
                          abs(bxy[1] - BOX_POS[1]) < 0.07 and
                          bz < BOX_WALL_TOP_Z + 0.02)
                if on_table and not in_box:
                    objects_on_table.append(bname)
            print(f"\n    Objects remaining on table: {len(objects_on_table)} — {objects_on_table}")
            if len(objects_on_table) == 0:
                print("    All objects placed in box! Done.")
                break

            # === Step 4: Capture RGB-D ===
            print("[4] Capturing RGB-D from D435i...")
            mujoco.mj_forward(robot.model, robot.data)
            rgb, depth = render_rgbd(robot.model, robot.data, cfgs.width, cfgs.height)
            print(f"    RGB: {rgb.shape}, Depth range: [{depth[depth>0].min():.3f}, {depth[depth>0].max():.3f}] m")

            # === Step 5: Create workspace mask (center region) ===
            workspace_mask = np.zeros((cfgs.height, cfgs.width), dtype=bool)
            margin_x, margin_y = cfgs.width // 8, cfgs.height // 8
            workspace_mask[margin_y:cfgs.height-margin_y,
                          margin_x:cfgs.width-margin_x] = True

            # === Step 6: GraspNet prediction ===
            print("[5] Running GraspNet inference...")
            end_points, cloud_o3d = graspnet_infer.process_realsense_data(
                rgb, depth, fx, fy, cx, cy, workspace_mask
            )
            target_gg = graspnet_infer.predict_grasps(end_points, cloud_o3d)

            if len(target_gg) == 0:
                print("    No valid grasps found. Try again.")
                continue

            best_grasp = target_gg[0]
            print(f"    Best grasp: score={best_grasp.score:.4f}, "
                  f"width={best_grasp.width:.4f}, depth={best_grasp.depth:.4f}")
            print(f"    Total grasps: {len(target_gg)}")

            # === Step 6b: Optional visualization ===
            if cfgs.vis_grasps:
                import open3d as o3d
                grippers_vis = target_gg.to_open3d_geometry_list()
                # Flip Z for better viewing
                T_flip = np.diag([1, 1, -1, 1])
                cloud_vis = o3d.geometry.PointCloud(cloud_o3d)
                cloud_vis.transform(T_flip)
                for g in grippers_vis:
                    g.transform(T_flip)
                o3d.visualization.draw_geometries([cloud_vis, *grippers_vis])

            # === Step 7: Detect objects from depth and compute grasp target ===
            print("[6] Detecting objects from depth point cloud...")
            obj_centroids = detect_objects_from_depth(
                depth, robot.model, robot.data, "d435i_depth",
                cfgs.width, cfgs.height, table_z=TABLE_Z,
                exclude_regions=[(BOX_POS[0], BOX_POS[1], BOX_EXCLUDE_RADIUS)]
            )

            if len(obj_centroids) == 0:
                print("    No objects detected above table. Try again.")
                continue

            # Use the nearest object centroid as the grasp target
            target_centroid = obj_centroids[0]
            print(f"    Detected {len(obj_centroids)} object(s)")
            print(f"    Target centroid: {target_centroid}")

            # Grasp position: centroid XY, estimate object center Z from visible surface
            # Visible centroid_z ≈ object top; center ≈ midpoint of table and top
            grasp_z = (TABLE_Z + target_centroid[2]) / 2.0
            grasp_pos = np.array([target_centroid[0], target_centroid[1], grasp_z])
            print(f"    Grasp target: {grasp_pos}")

            # === Step 8: Execute grasp via IK ===
            print("[7] Executing grasp...")
            QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # 180° about X → pointing straight down

            # Pre-grasp: 10cm above grasp point
            pre_grasp_pos = grasp_pos.copy()
            pre_grasp_pos[2] += 0.10

            # Open gripper and move above grasp
            robot.gripper_open(duration=0.3)
            robot.move_to_ee_position(pre_grasp_pos, duration=3.0, target_quat=QUAT_DOWN)
            print(f"    Pre-grasp EE: {robot.ee_position}")

            # Move down to grasp position
            robot.move_to_ee_position(grasp_pos, duration=2.0, target_quat=QUAT_DOWN)
            print(f"    Grasp EE: {robot.ee_position}")
            print(f"    Position error: {np.linalg.norm(robot.ee_position - grasp_pos):.4f}")

            # Close gripper
            robot.gripper_close(duration=1.0)
            print(f"    Gripper width: {robot.gripper_width:.4f}")

            # Lift up
            lift_pos = robot.ee_position.copy()
            lift_pos[2] += 0.15
            robot.move_to_ee_position(lift_pos, duration=2.0, target_quat=QUAT_DOWN)
            print(f"    Lift EE: {robot.ee_position}")

            # Check if object is grasped — find which body was lifted
            grasped_body = None
            for bname in GRASPABLE_BODIES:
                bz = robot.data.body(bname).xpos[2]
                if bz > TABLE_Z + 0.05:
                    grasped_body = bname
                    break

            if grasped_body:
                print(f"    SUCCESS! Grasped '{grasped_body}' (z={robot.data.body(grasped_body).xpos[2]:.3f})")
            else:
                print(f"    No object lifted — grasp may have failed")
                robot.gripper_open(duration=0.5)
                robot.move_to_joint_position(HOME, duration=2.0)
                print("    Cycle complete (grasp failed).\n")
                continue

            # === Step 9: Place into box ===
            print(f"[8] Placing '{grasped_body}' into box...")
            BOX_INNER_Z = BOX_POS[2] + 0.006 + 0.015  # bottom top + lower release height

            # Move above box
            above_box = np.array([BOX_POS[0], BOX_POS[1], BOX_WALL_TOP_Z + 0.08])
            robot.move_to_ee_position(above_box, duration=2.5, target_quat=QUAT_DOWN)
            print(f"    Above box EE: {robot.ee_position}")

            # Lower into box
            place_pos = np.array([BOX_POS[0], BOX_POS[1], BOX_INNER_Z])
            robot.move_to_ee_position(place_pos, duration=2.0, target_quat=QUAT_DOWN)
            print(f"    Place EE: {robot.ee_position}")

            # Release
            robot.gripper_open(duration=0.5)
            print(f"    Released. Gripper width: {robot.gripper_width:.4f}")

            # Retract upward
            retract_pos = robot.ee_position.copy()
            retract_pos[2] = BOX_WALL_TOP_Z + 0.10
            robot.move_to_ee_position(retract_pos, duration=1.5, target_quat=QUAT_DOWN)

            # Check result
            obj_final = robot.data.body(grasped_body).xpos.copy()
            in_box = (
                abs(obj_final[0] - BOX_POS[0]) < 0.07 and
                abs(obj_final[1] - BOX_POS[1]) < 0.07 and
                obj_final[2] > TABLE_Z and obj_final[2] < BOX_WALL_TOP_Z + 0.02
            )
            if in_box:
                print(f"    PLACED! '{grasped_body}' at {obj_final} — inside box")
            else:
                print(f"    '{grasped_body}' at {obj_final} — may have missed the box")

            # Return home
            robot.move_to_joint_position(HOME, duration=2.0)
            print("    Cycle complete.\n")

    except KeyboardInterrupt:
        print("\nExiting...")
    except EOFError:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
