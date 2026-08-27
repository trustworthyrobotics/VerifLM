import argparse, time
from collections import deque
from pathlib import Path

import mujoco, mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import SceneBuilder
from src.trajectory import GraspPlanner, TrajectoryPlanner, EEF_BODY_NAME

POS_STEP, ROT_STEP = 0.015, 5.0
parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
parser.add_argument("--stage", type=int, default=0)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
NAME, STAGE = args.benchmark, args.stage
CONFIG = BENCHMARK_CONFIGS[NAME]
if not 0 <= STAGE < len(CONFIG["active"]): raise ValueError(f"stage must be in [0,{len(CONFIG['active'])-1}]")
ACTIVE_NAME, PASSIVE_NAME = CONFIG["active"][STAGE], CONFIG["passive"][STAGE]
ACTIVE = OBJECT_CATALOG[ACTIVE_NAME]
ART = ROOT/"artifacts"/f"{NAME}_stage{STAGE}"; PER, TRAJ = ART/"perception", ART/"trajectory"
XML = ROOT/"models/generated_xmls"; TRAJ.mkdir(parents=True,exist_ok=True); XML.mkdir(parents=True,exist_ok=True)

def previous_obstacles():
    obs = []
    for s in range(STAGE):
        name = CONFIG["active"][s]
        if name in (ACTIVE_NAME,PASSIVE_NAME) or any(name == n for n,_ in obs): continue
        p = ROOT/"artifacts"/f"{NAME}_stage{s}"/"dtwin"/"T_mujoco_active_goal_selected.npy"
        if not p.exists(): raise FileNotFoundError(f"Missing previous-stage goal pose: {p}")
        obs.append((name,np.load(p)))
    return obs

T_cam_board = np.load(ROOT/"artifacts/calibration/extrinsics.npy")
Ta, Tp = np.load(PER/"source_active_pose_cam.npy"), np.load(PER/"source_passive_pose_cam.npy")
T_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Ta
T_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Tp

scene = SceneBuilder().build(ACTIVE_NAME,PASSIVE_NAME,T_active,T_passive,T_active,
    obstacles=previous_obstacles(), output_path=XML/f"test_{NAME}_stage{STAGE}_trajectory.xml")
grasps = GraspPlanner(ACTIVE,None,None).load_hardcoded()
if not grasps: raise RuntimeError(f"{ACTIVE_NAME} has no hardcoded grasps.")

data = mujoco.MjData(scene.model); base_qpos = scene.data.qpos.copy()
active_joint = int(scene.model.body_jntadr[scene.active_body_id])
active_qadr, active_dadr = int(scene.model.jnt_qposadr[active_joint]), int(scene.model.jnt_dofadr[active_joint])

state = {"mode":"edit","target":"goal","source":T_active.copy(),"goal":T_active.copy(),"plan":False,"reset":False}
keys = deque(); planner = trajectory = None
stage_i = point_i = 0; stage_start = None; attached = False

def set_active(d,T):
    q = Rotation.from_matrix(T[:3,:3]).as_quat()
    d.qpos[active_qadr:active_qadr+3] = T[:3,3]; d.qpos[active_qadr+3:active_qadr+7] = q[[3,0,1,2]]
    d.qvel[active_dadr:active_dadr+6] = 0

def set_mocap(d,mocap_id,T):
    q = Rotation.from_matrix(T[:3,:3]).as_quat()
    d.mocap_pos[mocap_id] = T[:3,3]; d.mocap_quat[mocap_id] = q[[3,0,1,2]]

def reset_view():
    data.qpos[:] = base_qpos; data.qvel[:] = 0
    set_active(data,state["source"]); set_mocap(data,scene.goal_mocap_id,state["source"])
    set_mocap(data,scene.refined_goal_mocap_id,state["goal"]); mujoco.mj_forward(scene.model,data)

def prepare_planner():
    scene.data.qpos[:] = base_qpos; scene.data.qvel[:] = 0
    set_active(scene.data,state["source"]); set_mocap(scene.data,scene.goal_mocap_id,state["source"])
    set_mocap(scene.data,scene.refined_goal_mocap_id,state["goal"]); mujoco.mj_forward(scene.model,scene.data)

def move_target(dp=None,axis=None,deg=0):
    T = state[state["target"]].copy()
    if dp is not None: T[:3,3] += np.asarray(dp)
    else: T[:3,:3] = T[:3,:3] @ Rotation.from_euler(axis,deg,degrees=True).as_matrix()
    state[state["target"]] = T
    if state["target"] == "source":
        set_active(data,T); set_mocap(data,scene.goal_mocap_id,T)
    else: set_mocap(data,scene.refined_goal_mocap_id,T)

def key_callback(key): keys.append(key)

def handle_key(key):
    if key == ord("R"): state["reset"] = True; return
    if state["mode"] != "edit": return
    if key == ord("1"): state["target"]="source"; print("[test]: editing SOURCE"); return
    if key == ord("2"): state["target"]="goal"; print("[test]: editing GOAL"); return
    moves = {263:[-POS_STEP,0,0],262:[POS_STEP,0,0],264:[0,-POS_STEP,0],265:[0,POS_STEP,0],267:[0,0,-POS_STEP],266:[0,0,POS_STEP]}
    rots = {ord("Q"):("x",ROT_STEP),ord("A"):("x",-ROT_STEP),ord("W"):("y",ROT_STEP),ord("S"):("y",-ROT_STEP),ord("E"):("z",ROT_STEP),ord("D"):("z",-ROT_STEP)}
    if key in moves: move_target(moves[key])
    elif key in rots: move_target(axis=rots[key][0],deg=rots[key][1])
    elif key in (257,335): state["plan"] = True

def attach_object():
    eef = data.body(EEF_BODY_NAME); T = np.eye(4); T[:3,3],T[:3,:3] = eef.xpos,eef.xmat.reshape(3,3)
    set_active(data,T @ np.linalg.inv(planner.chosen_grasp.T_object_eef))

with mujoco.viewer.launch_passive(scene.model,data,key_callback=key_callback) as viewer:
    reset_view()
    print("1=SOURCE | 2=GOAL | Arrows=XY | PgUp/PgDn=Z | Q/A W/S E/D=RPY | Enter=plan | R=reset/edit | Ctrl+C=exit")
    print("[test]: green=source/unrefined goal, yellow=edited/refined goal")

    try:
        while viewer.is_running():
            while keys: handle_key(keys.popleft())

            try:
                if state["reset"]:
                    state["reset"]=False; state["mode"]="edit"; trajectory=None; reset_view()
                    print(f"[test]: reset; editing {state['target'].upper()}")

                if state["plan"]:
                    state["plan"]=False; reset_view(); prepare_planner(); planner=TrajectoryPlanner(scene)
                    trajectory = planner.plan(np.zeros(7),state["source"],state["goal"],grasps,
                        output_path=TRAJ/"test_trajectory.json")
                    if not trajectory:
                        state["mode"]="edit"; reset_view()
                        print("[test]: planning failed; adjust poses and press Enter again.")
                    else:
                        np.save(TRAJ/"test_source.npy",state["source"]); np.save(TRAJ/"test_goal.npy",state["goal"])
                        state["mode"]="replay"; stage_i=point_i=0; stage_start=None; attached=False; reset_view()
                        print("[test]: replaying continuously; R=reset/edit")

                if state["mode"]=="replay" and trajectory:
                    if stage_i >= len(trajectory):
                        reset_view(); stage_i=point_i=0; stage_start=None; attached=False
                    stage = trajectory[stage_i]
                    if stage.type=="grasp":
                        data.qpos[planner.finger_qadrs]=0.025; attached=True; stage_i+=1; stage_start=None
                    elif stage.type=="release":
                        attached=False; data.qpos[planner.finger_qadrs]=0.0; stage_i+=1; stage_start=None
                    else:
                        if stage_start is None: stage_start=time.monotonic(); point_i=0
                        elapsed=(time.monotonic()-stage_start)*3
                        while point_i < len(stage.q_path) and elapsed >= stage.time_from_start[point_i]:
                            data.qpos[planner.arm_qadrs]=stage.q_path[point_i]; mujoco.mj_forward(scene.model,data)
                            if attached: attach_object()
                            point_i += 1
                        if point_i >= len(stage.q_path): stage_i+=1; stage_start=None

                mujoco.mj_forward(scene.model,data)
                viewer.sync()
                time.sleep(1/120)

            except Exception as e:
                fatal_error = e
                print(f"[test]: error: {e}")
                break

    except KeyboardInterrupt:
        print("\n[test]: exiting...")

# Viewer is closed before propagating errors.
if fatal_error is not None:
    raise fatal_error

print("[test]: exited cleanly.")