#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import mujoco
import numpy as np

from src.config import BENCHMARK_CONFIGS
from src.digital_twin import Scene
from src.trajectory import Grasp, TrajectoryStage, TrajectoryPlanner

# REPLAY_SPEEDUP = 3.0   # trajectory.py currently does: t /= 3.0

parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()

root = Path(__file__).resolve().parent.parent
benchmark = args.benchmark
traj_path = root / "artifacts" / benchmark / "trajectory" / "trajectory.json"
xml_path = root / "models" / "generated_xmls" / f"{benchmark}.xml"

raw = json.loads(traj_path.read_text())
cfg = BENCHMARK_CONFIGS[benchmark]

# Load generated MuJoCo scene.
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
active_body_id = model.body(f"active_{cfg['active']}").id
passive_body_id = model.body(f"passive_{cfg['passive']}").id
scene = Scene(
    model=model,
    data=data,
    active_body_id=active_body_id,
    passive_body_id=passive_body_id,
    goal_mocap_id=int(model.body(f"goal_{cfg['active']}").mocapid[0]),
    table_geom_ids={model.geom(n).id for n in ("table", "table_base")},
    xml_path=xml_path,
)
mujoco.mj_forward(model, data)

# Reconstruct trajectory.
trajectory = [
    TrajectoryStage(
        s["type"],
        [np.asarray(q) for q in s["q_path"]] if s["q_path"] else None,
        s["time_from_start"],
    )
    for s in raw["trajectory"]
]

# Analyze all move waypoints.
all_q = np.vstack([s.q_path for s in trajectory if s.q_path])
min_q = np.min(all_q, axis=0)
max_q = np.max(all_q, axis=0)
max_vel = np.zeros(7)

for s in trajectory:
    if not s.q_path or len(s.q_path) < 2:
        continue

    q = np.asarray(s.q_path)
    t = np.asarray(s.time_from_start)
    v = np.abs(np.diff(q, axis=0) / np.diff(t)[:, None])
    max_vel = np.maximum(max_vel, v.max(axis=0))

print("\nJoint   Min q [deg]   Max q [deg]   Max velocity [rad/s]")
for i in range(7):
    print(
        f"J{i+1:<2}     "
        f"{np.rad2deg(min_q[i]):10.4f}   "
        f"{np.rad2deg(max_q[i]):10.4f}   "
        f"{(max_vel[i]):10.4f}"
    )

print(f"\nOverall min q:       {np.rad2deg(min_q.min()):.4f} deg")
print(f"Overall max q:       {np.rad2deg(max_q.max()):.4f} deg")
print(f"Overall max velocity:{(max_vel.max()):.4f} rad/s\n")


# Replay.
planner = TrajectoryPlanner(scene)
planner.chosen_grasp = Grasp(
    np.asarray(raw["T_object_eef"]),
    np.asarray(raw["T_object_pregrasp_eef"]),
)
planner.replay_trajectory(trajectory)