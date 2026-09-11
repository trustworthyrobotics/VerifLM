#!/usr/bin/env python3
"""Inspect and replay a saved trajectory from a standard or estimated run."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from src.config import BENCHMARK_CONFIGS
from src.digital_twin import Scene
from src.trajectory import Grasp, TrajectoryPlanner, TrajectoryStage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--estimated", action="store_true")
    parser.add_argument("--stage", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    benchmark = args.benchmark

    if benchmark not in BENCHMARK_CONFIGS:
        parser.error(f"unknown benchmark: {benchmark}")

    cfg = BENCHMARK_CONFIGS[benchmark]
    if not 0 <= args.stage < len(cfg["active"]):
        parser.error(f"--stage must be in [0, {len(cfg['active']) - 1}]")

    active_name = cfg["active"][args.stage]
    passive_name = cfg["passive"][args.stage]

    run_dir = root / "artifacts" / f"{benchmark}_stage{args.stage}"
    xml_suffix = ""
    if args.estimated:
        run_dir /= "estimated"
        xml_suffix = "_estimated"

    filename = "test_trajectory.json" if args.test else "trajectory.json"
    traj_path = run_dir / "trajectory" / filename
    xml_path = (
        root
        / "models"
        / "generated_xmls"
        / f"{benchmark}_stage{args.stage}{xml_suffix}.xml"
    )

    for label, path in (("trajectory", traj_path), ("scene XML", xml_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print(f"[replay]: trajectory: {traj_path}")
    print(f"[replay]: XML:        {xml_path}")

    raw = json.loads(traj_path.read_text())

    # Load the matching generated MuJoCo scene.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    active_body_id = model.body(f"active_{active_name}").id
    passive_body_id = model.body(f"passive_{passive_name}").id
    obstacle_body_ids = {
        i
        for i in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(
            "obstacle_"
        )
    }

    scene = Scene(
        model=model,
        data=data,
        active_body_id=active_body_id,
        passive_body_id=passive_body_id,
        obstacle_body_ids=obstacle_body_ids,
        goal_mocap_id=int(model.body(f"goal_{active_name}").mocapid[0]),
        refined_goal_mocap_id=int(
            model.body(f"refined_goal_{active_name}").mocapid[0]
        ),
        table_geom_ids={model.geom(name).id for name in ("table", "table_base")},
        xml_path=xml_path,
    )
    mujoco.mj_forward(model, data)

    # Reconstruct the saved trajectory.
    trajectory = [
        TrajectoryStage(
            stage["type"],
            [np.asarray(q) for q in stage["q_path"]]
            if stage["q_path"]
            else None,
            stage["time_from_start"],
        )
        for stage in raw["trajectory"]
    ]

    move_paths = [stage.q_path for stage in trajectory if stage.q_path]
    if not move_paths:
        raise RuntimeError(f"trajectory contains no move waypoints: {traj_path}")

    # Analyze all move waypoints.
    all_q = np.vstack(move_paths)
    min_q = np.min(all_q, axis=0)
    max_q = np.max(all_q, axis=0)
    max_vel = np.zeros(7)

    for stage in trajectory:
        if not stage.q_path or len(stage.q_path) < 2:
            continue

        q = np.asarray(stage.q_path)
        t = np.asarray(stage.time_from_start)
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("trajectory contains non-increasing waypoint times")
        velocity = np.abs(np.diff(q, axis=0) / dt[:, None])
        max_vel = np.maximum(max_vel, velocity.max(axis=0))

    print("\nJoint   Min q [deg]   Max q [deg]   Max velocity [rad/s]")
    for i in range(7):
        print(
            f"J{i + 1:<2}     "
            f"{np.rad2deg(min_q[i]):10.4f}   "
            f"{np.rad2deg(max_q[i]):10.4f}   "
            f"{max_vel[i]:10.4f}"
        )

    print(f"\nOverall min q:        {np.rad2deg(min_q.min()):.4f} deg")
    print(f"Overall max q:        {np.rad2deg(max_q.max()):.4f} deg")
    print(f"Overall max velocity: {max_vel.max():.4f} rad/s\n")

    # replay_trajectory needs the grasp selected when the trajectory was planned.
    planner = TrajectoryPlanner(scene)
    planner.chosen_grasp = Grasp(
        np.asarray(raw["T_object_eef"]),
        np.asarray(raw["T_object_pregrasp_eef"]),
    )
    planner.replay_trajectory(trajectory)


if __name__ == "__main__":
    main()