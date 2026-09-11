"""Interactively adjust a saved source pose and test trajectory planning."""

import argparse
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from src.config import BENCHMARK_CONFIGS, OBJECT_CATALOG
from src.digital_twin import Scene
from src.trajectory import GraspPlanner, TrajectoryPlanner

POS_STEP = 0.015
ROT_STEP = 5.0


def pose_from_body(data, body_id):
    T = np.eye(4)
    T[:3, 3] = data.xpos[body_id]
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return T


def set_freejoint_pose(scene, T):
    joint = int(scene.model.body_jntadr[scene.active_body_id])
    qadr = int(scene.model.jnt_qposadr[joint])
    dadr = int(scene.model.jnt_dofadr[joint])
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    scene.data.qpos[qadr:qadr + 3] = T[:3, 3]
    scene.data.qpos[qadr + 3:qadr + 7] = quat[[3, 0, 1, 2]]
    scene.data.qvel[dadr:dadr + 6] = 0
    mujoco.mj_forward(scene.model, scene.data)


def set_mocap_pose(data, mocap_id, T):
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    data.mocap_pos[mocap_id] = T[:3, 3]
    data.mocap_quat[mocap_id] = quat[[3, 0, 1, 2]]


def load_scene(xml_path, active_name, passive_name):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    active_id = model.body(f"active_{active_name}").id
    passive_id = model.body(f"passive_{passive_name}").id
    obstacle_ids = {
        i for i in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("obstacle_")
    }
    scene = Scene(
        model=model,
        data=data,
        active_body_id=active_id,
        passive_body_id=passive_id,
        obstacle_body_ids=obstacle_ids,
        goal_mocap_id=int(model.body(f"goal_{active_name}").mocapid[0]),
        refined_goal_mocap_id=int(model.body(f"refined_goal_{active_name}").mocapid[0]),
        table_geom_ids={model.geom(name).id for name in ("table", "table_base")},
        xml_path=xml_path,
    )
    mujoco.mj_forward(model, data)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--estimated", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = BENCHMARK_CONFIGS[args.benchmark]
    PREINSERTION = config["preinsertion"][args.stage] if "preinsertion" in config else None
    if not 0 <= args.stage < len(config["active"]):
        parser.error(f"--stage must be in [0, {len(config['active']) - 1}]")

    active_name = config["active"][args.stage]
    passive_name = config["passive"][args.stage]
    suffix = "_estimated" if args.estimated else ""
    run_dir = root / "artifacts" / f"{args.benchmark}_stage{args.stage}"
    if args.estimated:
        run_dir /= "estimated"

    xml_path = root / "models" / "generated_xmls" / f"{args.benchmark}_stage{args.stage}{suffix}.xml"
    goal_path = run_dir / "dtwin" / "T_mujoco_active_goal_selected.npy"
    output_path = run_dir / "trajectory" / "test_trajectory.json"
    for path in (xml_path, goal_path):
        if not path.exists():
            raise FileNotFoundError(path)

    scene = load_scene(xml_path, active_name, passive_name)
    source = pose_from_body(scene.data, scene.active_body_id)
    goal = np.load(goal_path)
    set_mocap_pose(scene.data, scene.refined_goal_mocap_id, goal)
    set_freejoint_pose(scene, source)

    grasps = GraspPlanner(OBJECT_CATALOG[active_name], None, None).load_hardcoded()
    if not grasps:
        raise RuntimeError(f"{active_name} has no hardcoded grasps")

    keys = deque()
    mode = "position"
    target = "source"
    plan_requested = False
    replay_planner = None
    replay_trajectory = None

    def key_callback(key):
        keys.append(key)

    print(f"[test]: XML:  {xml_path}")
    print(f"[test]: goal: {goal_path}")
    print("[test]: SOURCE POSITION | \\=source/goal | /=position/orientation")
    print("[test]: arrows=XY or roll/pitch | PgUp/PgDn=Z or yaw | Enter=plan")

    with mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            while keys:
                key = keys.popleft()
                if key == ord("/"):
                    mode = "orientation" if mode == "position" else "position"
                    print(f"[test]: {target.upper()} {mode.upper()}")
                    continue
                if key == ord("\\"):
                    target = "goal" if target == "source" else "source"
                    print(f"[test]: {target.upper()} {mode.upper()}")
                    continue
                if key in (257, 335):
                    plan_requested = True
                    continue

                signs = {263: -1, 262: 1, 264: -1, 265: 1, 267: -1, 266: 1}
                if key not in signs:
                    continue
                sign = signs[key]
                T = source if target == "source" else goal
                if mode == "position":
                    axis = {263: 0, 262: 0, 264: 1, 265: 1, 267: 2, 266: 2}[key]
                    T[axis, 3] += sign * POS_STEP
                else:
                    axis = {263: "x", 262: "x", 264: "y", 265: "y", 267: "z", 266: "z"}[key]
                    T[:3, :3] = T[:3, :3] @ Rotation.from_euler(
                        axis, sign * ROT_STEP, degrees=True
                    ).as_matrix()
                if target == "source":
                    set_freejoint_pose(scene, source)
                else:
                    set_mocap_pose(scene.data, scene.refined_goal_mocap_id, goal)
                    mujoco.mj_forward(scene.model, scene.data)

            if plan_requested:
                plan_requested = False
                output_path.parent.mkdir(parents=True, exist_ok=True)
                planner = TrajectoryPlanner(scene)
                trajectory = planner.plan(
                    q_start=np.zeros(7),
                    T_mujoco_object_src=source,
                    T_mujoco_object_goal=goal,
                    grasps=grasps,
                    output_path=output_path,
                    preinsertion_offset=PREINSERTION
                )
                if trajectory:
                    print(f"[test]: saved {output_path}")
                    replay_planner = planner
                    replay_trajectory = trajectory
                    viewer.close()
                    break
                print("[test]: planning failed")

            viewer.sync()
            time.sleep(1 / 120)

    if replay_trajectory is not None:
        print("[test]: replaying successful trajectory")
        replay_planner.replay_trajectory(replay_trajectory)


if __name__ == "__main__":
    main()
