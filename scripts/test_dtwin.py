import argparse
import time
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
Ta = np.load(PERCEPTION_DIR / "source_active_pose_cam.npy")
Tp = np.load(PERCEPTION_DIR / "source_passive_pose_cam.npy")
Ta_goal = np.load(PERCEPTION_DIR / "edited_active_pose_cam.npy")
Tp_goal = np.load(PERCEPTION_DIR / "edited_passive_pose_cam.npy")

T_mujoco_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Ta
T_mujoco_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Tp
T_mujoco_active_goal = T_mujoco_passive @ np.linalg.inv(Tp_goal) @ Ta_goal

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
    CONFIG["active"], CONFIG["passive"],
    T_mujoco_active, T_mujoco_passive, T_mujoco_active_goal,
    ROOT / "models/generated_xmls" / f"{NAME}.xml",
)
digital_twin = DigitalTwin(scene, V, S)

# Sampling viewer + recording
input("Press Enter to start sampling...")
candidates = digital_twin.sample_and_validate(
    n_samples=1000,
    max_valid=5,
    sim_steps=500,
    visualize=True,
    video_path=DTWIN_DIR / "test_sampling.mp4",
)

if not candidates:
    raise RuntimeError("No valid candidates found.")

for i, T in enumerate(candidates):
    np.save(DTWIN_DIR / f"valid_candidate_{i}.npy", T)

# Candidate-selection viewer: N/P cycle, Enter selects
data = mujoco.MjData(scene.model)
state = {"i": 0, "changed": True, "selected": False}


def key_callback(key):
    if key == ord("N"):
        state["i"] = (state["i"] + 1) % len(candidates)
        state["changed"] = True
    elif key == ord("P"):
        state["i"] = (state["i"] - 1) % len(candidates)
        state["changed"] = True
    elif key in (257, 335):
        state["selected"] = True


with mujoco.viewer.launch_passive(scene.model, data, key_callback=key_callback) as viewer:
    print("N = next, P = previous, Enter = select")

    while viewer.is_running() and not state["selected"]:
        if state["changed"]:
            i = state["i"]
            data.qpos[:] = scene.data.qpos
            digital_twin.set_body_pose(
                data, digital_twin.active_qadr, digital_twin.active_dadr, candidates[i]
            )
            print(f"Candidate {i + 1}/{len(candidates)}")
            state["changed"] = False

        viewer.sync()
        time.sleep(0.01)

if not state["selected"]:
    raise RuntimeError("Viewer closed before selection.")

selected = candidates[state["i"]]
np.save(DTWIN_DIR / "selected_candidate.npy", selected)

# Trajectory planning
grasps = GraspPlanner(ACTIVE, None, None).load_hardcoded()
planner = TrajectoryPlanner(scene)

trajectory = planner.plan(
    q_start=np.zeros(7),
    T_mujoco_object_src=T_mujoco_active,
    T_mujoco_object_goal=selected,
    grasps=grasps,
    output_path=TRAJECTORY_DIR / "test_trajectory.json",
)

if not trajectory:
    raise RuntimeError("Trajectory planning failed.")

planner.replay_trajectory(trajectory)