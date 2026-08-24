import argparse, signal, time
from collections import deque
from pathlib import Path

import mujoco, mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import SceneBuilder
from src.trajectory import GraspPlanner, TrajectoryPlanner

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
ART, XML = ROOT/"artifacts"/f"{NAME}_stage{STAGE}", ROOT/"models/generated_xmls"
PER = ART/"perception"; XML.mkdir(parents=True, exist_ok=True)

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
T_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Ta
T_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Tp

scene = SceneBuilder().build(ACTIVE_NAME, PASSIVE_NAME, T_active, T_passive, T_active,
    obstacles=previous_obstacles(), output_path=XML/f"test_{NAME}_stage{STAGE}_grasps.xml")
model, data = scene.model, mujoco.MjData(scene.model)
base_qpos = scene.data.qpos.copy()
active_joint = int(model.body_jntadr[scene.active_body_id])
active_qadr, active_dadr = int(model.jnt_qposadr[active_joint]), int(model.jnt_dofadr[active_joint])

grasps = GraspPlanner(ACTIVE,None,None).load_hardcoded()
if not grasps: raise RuntimeError(f"{ACTIVE_NAME} has no hardcoded grasps.")
planner = TrajectoryPlanner(scene)
state, keys = {"source":T_active.copy(),"grasp_i":None,"stop":False}, deque()

def set_active(d,T):
    q = Rotation.from_matrix(T[:3,:3]).as_quat()
    d.qpos[active_qadr:active_qadr+3] = T[:3,3]; d.qpos[active_qadr+3:active_qadr+7] = q[[3,0,1,2]]
    d.qvel[active_dadr:active_dadr+6] = 0

def set_mocap(d,mocap_id,T):
    q = Rotation.from_matrix(T[:3,:3]).as_quat()
    d.mocap_pos[mocap_id] = T[:3,3]; d.mocap_quat[mocap_id] = q[[3,0,1,2]]

def reset_robot():
    data.qpos[:] = base_qpos; data.qvel[:] = 0
    set_active(data,state["source"]); set_mocap(data,scene.goal_mocap_id,state["source"])
    set_mocap(data,scene.refined_goal_mocap_id,state["source"]); mujoco.mj_forward(model,data)

def move_source(dp=None,axis=None,deg=0):
    T = state["source"].copy()
    if dp is not None: T[:3,3] += np.asarray(dp,dtype=float)
    else: T[:3,:3] = T[:3,:3] @ Rotation.from_euler(axis,deg,degrees=True).as_matrix()
    state["source"], state["grasp_i"] = T, None; reset_robot()

def show_grasp(i):
    g = grasps[i]; Tpre, Tg = state["source"]@g.T_object_pregrasp_eef, state["source"]@g.T_object_eef
    planner.ik_data.qpos[:] = base_qpos; set_active(planner.ik_data,state["source"]); mujoco.mj_forward(model,planner.ik_data)
    qpre = planner.solve_ik(Tpre); qg = planner.solve_ik(Tg,qpre) if qpre is not None else None
    if qpre is None or qg is None:
        print(f"[test] grasp {i+1}/{len(grasps)} | pregrasp={'OK' if qpre is not None else 'FAIL'} | grasp={'FAIL' if qpre is not None else 'SKIP'}")
        reset_robot(); return
    data.qpos[:] = base_qpos; data.qvel[:] = 0; set_active(data,state["source"])
    set_mocap(data,scene.goal_mocap_id,state["source"]); set_mocap(data,scene.refined_goal_mocap_id,state["source"])
    data.qpos[planner.arm_qadrs] = qg; data.qpos[planner.finger_qadrs] = 0.025; mujoco.mj_forward(model,data)
    print(f"[test] grasp {i+1}/{len(grasps)} | pregrasp=OK | grasp=OK")

def key_callback(key): keys.append(key)
def handle_key(key):
    if key == ord("R"): state["source"],state["grasp_i"] = T_active.copy(),None; reset_robot(); print("[test]: source reset"); return
    moves = {263:[-POS_STEP,0,0],262:[POS_STEP,0,0],264:[0,-POS_STEP,0],265:[0,POS_STEP,0],267:[0,0,-POS_STEP],266:[0,0,POS_STEP]}
    rots = {ord("Q"):("x",ROT_STEP),ord("A"):("x",-ROT_STEP),ord("W"):("y",ROT_STEP),ord("S"):("y",-ROT_STEP),ord("E"):("z",ROT_STEP),ord("D"):("z",-ROT_STEP)}
    if key in moves: move_source(moves[key])
    elif key in rots: move_source(axis=rots[key][0],deg=rots[key][1])
    elif key in (ord("G"),ord("B")):
        step = 1 if key == ord("G") else -1
        state["grasp_i"] = (0 if step==1 else len(grasps)-1) if state["grasp_i"] is None else (state["grasp_i"]+step)%len(grasps)
        show_grasp(state["grasp_i"])

old = signal.getsignal(signal.SIGINT); signal.signal(signal.SIGINT,lambda *_: state.__setitem__("stop",True))
try:
    with mujoco.viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
        reset_robot()
        print("Arrows=XY | PgUp/PgDn=Z | Q/A W/S E/D=RPY | G/B=next/prev grasp | R=reset | Ctrl+C=exit")
        print(f"[test]: loaded {len(grasps)} grasps for {ACTIVE_NAME}")
        while viewer.is_running() and not state["stop"]:
            while keys: handle_key(keys.popleft())
            viewer.sync(); time.sleep(0.001)
finally: signal.signal(signal.SIGINT,old)
print("[test]: exited cleanly.")