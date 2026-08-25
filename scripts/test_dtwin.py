import argparse, time
from pathlib import Path

import mujoco, mujoco.viewer
import numpy as np

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import (DigitalTwin, SceneBuilder, SamplingConstraints, ValidationProgram,
    POSITION_CONSTRAINT_MAP, ORIENTATION_CONSTRAINT_MAP, POSITION_TOLERANCE_MAP, ORIENTATION_TOLERANCE_MAP)
from src.render import render_for_vlm
from src.trajectory import GraspPlanner, TrajectoryPlanner
from src.vlm import load_sampling_constraints, load_validation_program

parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--default-V", dest="default_v", action="store_true")
parser.add_argument("--default-S", dest="default_s", action="store_true")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
NAME, STAGE = args.benchmark, args.stage
CONFIG = BENCHMARK_CONFIGS[NAME]
if not 0 <= STAGE < len(CONFIG["active"]): raise ValueError(f"stage must be in [0,{len(CONFIG['active'])-1}]")
ACTIVE_NAME, PASSIVE_NAME = CONFIG["active"][STAGE], CONFIG["passive"][STAGE]
ACTIVE, PASSIVE = OBJECT_CATALOG[ACTIVE_NAME], OBJECT_CATALOG[PASSIVE_NAME]

ART = ROOT/"artifacts"/f"{NAME}_stage{STAGE}"
PER, VLM, DTWIN, RENDER, TRAJ = (ART/x for x in ("perception","vlm","dtwin","render","trajectory"))
XML = ROOT/"models/generated_xmls"
for d in (DTWIN, RENDER, TRAJ, XML): d.mkdir(parents=True, exist_ok=True)

def previous_obstacles():
    obs = []
    for s in range(STAGE):
        name = CONFIG["active"][s]
        if name in (ACTIVE_NAME, PASSIVE_NAME) or any(name == n for n, _ in obs): continue
        p = ROOT/"artifacts"/f"{NAME}_stage{s}"/"dtwin"/"T_mujoco_active_goal_selected.npy"
        if not p.exists(): raise FileNotFoundError(f"Missing previous-stage goal pose: {p}")
        obs.append((name, np.load(p)))
    return obs

T_cam_board = np.load(ROOT/"artifacts/calibration/extrinsics.npy")
Ta, Tp = np.load(PER/"source_active_pose_cam.npy"), np.load(PER/"source_passive_pose_cam.npy")
Ta_g, Tp_g = np.load(PER/"edited_active_pose_cam.npy"), np.load(PER/"edited_passive_pose_cam.npy")
T_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Ta
T_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Tp
T_goal = T_passive @ np.linalg.inv(Tp_g) @ Ta_g

if args.default_s:
    L = max(ACTIVE["dimensions"]); p = POSITION_CONSTRAINT_MAP["normal_search"]*L; r = ORIENTATION_CONSTRAINT_MAP["normal_search"]
    S = SamplingConstraints(p,p,p,r,r,r,"none","none","none")
else:
    S = load_sampling_constraints(VLM/"translation_constraints.json", VLM/"rotation_constraints.json", ACTIVE["dimensions"])

if args.default_v:
    L = max(ACTIVE["dimensions"]); p = POSITION_TOLERANCE_MAP["moderate"]*L; r = ORIENTATION_TOLERANCE_MAP["moderate"]
    V = ValidationProgram("settling", [], [], [], p,p,p,r,r,r)
else:
    V = load_validation_program(VLM/"validation_program.json", ACTIVE["dimensions"])

scene = SceneBuilder().build(ACTIVE_NAME, PASSIVE_NAME, T_active, T_passive, T_goal,
    obstacles=previous_obstacles(), output_path=XML/f"test_{NAME}_stage{STAGE}_dtwin.xml")
dtwin = DigitalTwin(scene, V, S)

input("Press Enter to start sampling...")
candidates = dtwin.sample_and_validate(n_samples=1000, max_valid=5, sim_steps=500, visualize=True,
    video_path=DTWIN/"test_sampling.mp4")
if not candidates: raise RuntimeError("No valid candidates found.")

view = CONFIG["render_config"][1]
for i, T in enumerate(candidates):
    np.save(DTWIN/f"test_valid_candidate_{i}.npy", T)
    render_for_vlm(ACTIVE["mesh"], PASSIVE["mesh"], RENDER/f"test_valid_candidate_{i}.png",
        T_passive_active=np.linalg.inv(T_passive)@T, rendered_frame="active", **view)

data = mujoco.MjData(scene.model)
state = {"i":0, "changed":True, "selected":False}
def key_callback(key):
    if key == ord("N"): state["i"], state["changed"] = (state["i"]+1)%len(candidates), True
    elif key == ord("P"): state["i"], state["changed"] = (state["i"]-1)%len(candidates), True
    elif key in (257,335): state["selected"] = True

with mujoco.viewer.launch_passive(scene.model, data, key_callback=key_callback) as viewer:
    print("N=next | P=previous | Enter=select")
    while viewer.is_running() and not state["selected"]:
        if state["changed"]:
            i = state["i"]; data.qpos[:] = scene.data.qpos; data.qvel[:] = 0
            data.mocap_pos[:] = scene.data.mocap_pos; data.mocap_quat[:] = scene.data.mocap_quat
            dtwin.set_body_pose(data, dtwin.active_qadr, dtwin.active_dadr, candidates[i])
            print(f"Candidate {i+1}/{len(candidates)}"); state["changed"] = False
        viewer.sync(); time.sleep(0.01)

if not state["selected"]: raise RuntimeError("Viewer closed before selection.")
selected = candidates[state["i"]]
np.save(DTWIN/"test_selected_candidate.npy", selected)
dtwin.set_mocap_pose(scene.data, scene.refined_goal_mocap_id, selected)

planner = TrajectoryPlanner(scene)
trajectory = planner.plan(np.zeros(7), T_active, selected, GraspPlanner(ACTIVE,None,None).load_hardcoded(),
    output_path=TRAJ/"test_trajectory.json")
if not trajectory: raise RuntimeError("Trajectory planning failed.")
planner.replay_trajectory(trajectory)