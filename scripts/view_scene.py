from __future__ import annotations

import argparse
import pathlib
import time

import mujoco
import mujoco.viewer


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "fr3_table_grasp.xml"


def load_model(xml_path: pathlib.Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
      mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
      mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the simplified FR3 tabletop grasp scene.")
    parser.add_argument("--xml", type=pathlib.Path, default=DEFAULT_XML, help="Path to a MuJoCo XML scene.")
    parser.add_argument("--camera", default="overview", help="Initial viewer camera name.")
    args = parser.parse_args()

    xml_path = args.xml.expanduser().resolve()
    model, data = load_model(xml_path)
    reset_home(model, data)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        # Use free camera so user can zoom/rotate/pan
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.1, 0.0, 0.85]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()