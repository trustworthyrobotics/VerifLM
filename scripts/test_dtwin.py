import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import DigitalTwin, SceneBuilder
from src.trajectory import GraspPlanner, TrajectoryPlanner
from src.vlm import load_sampling_constraints, load_validation_program


parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
NAME = args.benchmark
CONFIG = BENCHMARK_CONFIGS[NAME]
ACTIVE = OBJECT_CATALOG[CONFIG["active"]]

ARTIFACT_DIR = ROOT / "artifacts" / NAME
PERCEPTION_DIR = ARTIFACT_DIR / "perception"
VLM_DIR = ARTIFACT_DIR / "vlm"
DTWIN_DIR = ARTIFACT_DIR / "dtwin"
TRAJECTORY_DIR = ARTIFACT_DIR / "trajectory"

DTWIN_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

# Recover poses
T_cam_board = np.load(ROOT / "artifacts/calibration/extrinsics.npy")
T_cam_active = np.load(PERCEPTION_DIR / "source_active_pose_cam.npy")
T_cam_passive = np.load(PERCEPTION_DIR / "source_passive_pose_cam.npy")
T_cam_active_goal = np.load(PERCEPTION_DIR / "edited_active_pose_cam.npy")
T_cam_passive_goal = np.load(PERCEPTION_DIR / "edited_passive_pose_cam.npy")

T_mujoco_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_active
T_mujoco_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_passive
T_mujoco_active_goal = (
    T_mujoco_passive
    @ np.linalg.inv(T_cam_passive_goal)
    @ T_cam_active_goal
)

# Recover V and S
S = load_sampling_constraints(
    VLM_DIR / "translation_constraints.json",
    VLM_DIR / "rotation_constraints.json",
    ACTIVE["dimensions"],
)
V = load_validation_program(
    VLM_DIR / "validation_program.json",
    ACTIVE["dimensions"],
)

# Build digital twin
scene = SceneBuilder().build(
    CONFIG["active"],
    CONFIG["passive"],
    T_mujoco_active,
    T_mujoco_passive,
    T_mujoco_active_goal,
    ROOT / "models/generated_xmls" / f"{NAME}.xml",
)
digital_twin = DigitalTwin(scene, V, S)

input("Press Enter to start sampling...")

candidates = digital_twin.sample(
    n_samples=1000,
    max_valid=5,
    sim_steps=500,
    visualize=False,
)

if not candidates:
    raise RuntimeError("No valid candidates found.")

for i, T in enumerate(candidates):
    np.save(DTWIN_DIR / f"valid_candidate_{i}.npy", T)

# Browse candidates
data = mujoco.MjData(scene.model)

with mujoco.viewer.launch_passive(scene.model, data) as viewer:
    i = 0

    while viewer.is_running():
        data.qpos[:] = scene.data.qpos
        digital_twin.set_body_pose(
            data,
            digital_twin.active_qadr,
            digital_twin.active_dadr,
            candidates[i],
        )
        viewer.sync()

        choice = input(
            f"Candidate {i + 1}/{len(candidates)} "
            "[Enter]=select, n=next, p=previous: "
        ).strip().lower()

        if choice == "":
            selected = candidates[i]
            break
        elif choice == "n":
            i = (i + 1) % len(candidates)
        elif choice == "p":
            i = (i - 1) % len(candidates)
    else:
        raise RuntimeError("Viewer closed before selection.")

np.save(DTWIN_DIR / "selected_candidate.npy", selected)

# Trajectory planning
grasps = GraspPlanner(ACTIVE, None, None).load_hardcoded()
planner = TrajectoryPlanner(scene)

trajectory = planner.plan(
    q_start=np.zeros(7),
    T_mujoco_object_src=T_mujoco_active,
    T_mujoco_object_goal=selected,
    grasps=grasps,
    output_path=TRAJECTORY_DIR / "trajectory.json",
)

if not trajectory:
    raise RuntimeError("Trajectory planning failed.")

planner.replay_trajectory(trajectory)