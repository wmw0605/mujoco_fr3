"""Capture RGB and depth images from the simulated D435i eye-in-hand camera."""
from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "fr3_table_grasp.xml"

# D435i intrinsics (640x480)
WIDTH, HEIGHT = 640, 480
RGB_FOVY = 42.5   # degrees
DEPTH_FOVY = 58.0  # degrees
DEPTH_RANGE = (0.105, 10.0)  # meters


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RGB+depth from D435i wrist camera.")
    parser.add_argument("--xml", type=pathlib.Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "captures",
                        help="Output directory for saved images.")
    parser.add_argument("--show", action="store_true", help="Display images with matplotlib.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    # --- RGB ---
    renderer.update_scene(data, camera="d435i_rgb")
    rgb = renderer.render().copy()

    # --- Depth (MuJoCo Renderer returns linear depth in meters directly) ---
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="d435i_depth")
    metric_depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    # Clip to D435i effective range
    metric_depth = np.clip(metric_depth, DEPTH_RANGE[0], DEPTH_RANGE[1])

    print(f"RGB  : {rgb.shape}, dtype={rgb.dtype}")
    print(f"Depth: {metric_depth.shape}, range=[{metric_depth.min():.3f}, {metric_depth.max():.3f}] m")

    # Save
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = out_dir / "d435i_rgb.npy"
    depth_path = out_dir / "d435i_depth.npy"
    np.save(str(rgb_path), rgb)
    np.save(str(depth_path), metric_depth)
    print(f"Saved: {rgb_path}")
    print(f"Saved: {depth_path}")

    if args.show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("D435i RGB")
        axes[0].axis("off")
        im = axes[1].imshow(metric_depth, cmap="turbo")
        axes[1].set_title("D435i Depth (m)")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(str(out_dir / "d435i_preview.png"), dpi=120)
        print(f"Saved: {out_dir / 'd435i_preview.png'}")
        plt.show()


if __name__ == "__main__":
    main()
