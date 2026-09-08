"""Place in the repository root; run: python ensemble_teleop.py mug_on_rack."""
import argparse
import json
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from src.config import BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import DigitalTwin
from src.vlm import load_validation_program


def quat(T):
    return np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)


def twin_from_xml(path, active, passive, V):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = SimpleNamespace(model=model, data=data, active_body_id=model.body(f"active_{active}").id,
        passive_body_id=model.body(f"passive_{passive}").id,
        goal_mocap_id=int(model.body(f"goal_{active}").mocapid[0]),
        refined_goal_mocap_id=int(model.body(f"refined_goal_{active}").mocapid[0]),
        table_geom_ids={model.geom(n).id for n in ("table", "table_base")},
        obstacle_body_ids={i for i in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("obstacle_")})
    return DigitalTwin(scene, V, None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark")
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--step", type=float, default=0.002, help="translation increment in meters")
    p.add_argument("--angle", type=float, default=2, help="rotation increment in degrees")
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    cfg = BENCHMARK_CONFIGS[args.benchmark]
    active, passive = cfg["active"][args.stage], cfg["passive"][args.stage]
    folder = root / "artifacts" / f"{args.benchmark}_stage{args.stage}" / "estimated"
    load = lambda name: np.load(folder / "perception" / f"{name}_pose_cam.npy")
    Tp = T_mujoco_board @ np.linalg.inv(np.load(root/"artifacts/calibration/extrinsics.npy")) @ load("source_passive")
    Ta = Tp @ np.linalg.inv(load("edited_passive")) @ load("edited_active")
    poses, initial = [Ta.copy(), Tp.copy()], [Ta.copy(), Tp.copy()]
    dimensions = json.loads((root/"artifacts/object_estimation"/active/"estimate.json").read_text())["dimensions"]
    V = load_validation_program(folder/"vlm/validation_program.json", dimensions)
    prefix = root/"models/generated_xmls"/f"{args.benchmark}_stage{args.stage}_estimated"
    paths = [Path(f"{prefix}_world_{i}.xml") for i in range(100)]
    missing = [str(path) for path in [Path(f"{prefix}.xml"), *paths] if not path.exists()]
    if missing: raise FileNotFoundError("Run estimated-mode world construction first; missing: " + missing[0])
    # Make the preview passive body movable without editing the simulation worlds.
    tree = ET.parse(Path(f"{prefix}.xml"))
    body = tree.getroot().find(f".//body[@name='passive_{passive}']")
    body.set("mocap", "true")
    with tempfile.TemporaryDirectory(prefix="teleop_", dir=prefix.parent) as tmp:
        preview = Path(tmp)/"preview.xml"
        # Includes in generated XML are relative to its original directory.
        for elem in tree.getroot().iter():
            if elem.tag in ("include", "model") and elem.get("file"):
                elem.set("file", str((prefix.parent/elem.get("file")).resolve()))
        tree.write(preview)
        twin = twin_from_xml(preview, active, passive, V)
        m, d = twin.model, twin.scene.data
        pm = int(m.body(f"passive_{passive}").mocapid[0])
        for bid in (m.body(f"goal_{active}").id, m.body(f"refined_goal_{active}").id):
            m.geom_rgba[m.geom_bodyid == bid, 3] = 0
        keys, controlled, rotate = SimpleQueue(), 0, False
        print("T: active/passive | M: translate/rotate | arrows: X/Y or roll/pitch | PageUp/Down: Z or yaw | R: reset | Enter: score")
        with mujoco.viewer.launch_passive(m, d, key_callback=keys.put) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = Tp[:3, 3]
                viewer.cam.distance = 0.65
            while viewer.is_running():
                while not keys.empty():
                    key = keys.get()
                    if key == glfw.KEY_T: controlled = 1-controlled
                    elif key == glfw.KEY_BACKSLASH: rotate = not rotate
                    elif key == glfw.KEY_R: poses = [T.copy() for T in initial]
                    elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                        passed = []
                        print("Evaluating 100 worlds...")
                        for i, path in enumerate(paths):
                            world_xml = ET.parse(path)
                            b = world_xml.getroot().find(f".//body[@name='passive_{passive}']")
                            b.set("pos", " ".join(map(str, poses[1][:3, 3])))
                            b.set("quat", " ".join(map(str, quat(poses[1]))))
                            for elem in world_xml.getroot().iter():
                                if elem.tag in ("include", "model") and elem.get("file"):
                                    elem.set("file", str((path.parent/elem.get("file")).resolve()))
                            world_path = Path(tmp)/"world.xml"
                            world_xml.write(world_path)
                            world = twin_from_xml(world_path, active, passive, V)
                            passed.append(world.validate(poses[0].copy(), args.steps)[0])
                            del world
                            if (i+1) % 10 == 0:
                                print(f"  {i+1}/100 evaluated; {sum(passed)} passed")
                                viewer.sync()
                        print(f"ENSEMBLE: {sum(passed)}/100 passed ({sum(passed)}%)")
                    else:
                        move = {glfw.KEY_RIGHT:(0,1), glfw.KEY_LEFT:(0,-1), glfw.KEY_UP:(1,1),
                                glfw.KEY_DOWN:(1,-1), glfw.KEY_PAGE_UP:(2,1), glfw.KEY_PAGE_DOWN:(2,-1)}.get(key)
                        if move:
                            axis, sign = move
                            if rotate:
                                delta = np.zeros(3); delta[axis] = sign*np.deg2rad(args.angle)
                                poses[controlled][:3, :3] = Rotation.from_rotvec(delta).as_matrix() @ poses[controlled][:3, :3]
                            else: poses[controlled][axis, 3] += sign*args.step
                    if key in (glfw.KEY_T, glfw.KEY_BACKSLASH):
                        print(f"Control: {('active','passive')[controlled]} | {('translation','rotation')[rotate]} (world axes)")
                with viewer.lock():
                    twin.set_mocap_pose(d, pm, poses[1])
                    twin.set_body_pose(d, twin.active_qadr, twin.active_dadr, poses[0])
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    main()