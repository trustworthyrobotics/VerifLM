import argparse
import signal
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import SceneBuilder
from src.trajectory import GraspPlanner, TrajectoryPlanner, EEF_BODY_NAME


POS_STEP = 0.005
ROT_STEP = 5.0

parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
NAME = args.benchmark
CONFIG = BENCHMARK_CONFIGS[NAME]
ACTIVE = OBJECT_CATALOG[CONFIG["active"]]

ART = ROOT / "artifacts" / NAME
PER = ART / "perception"
TRAJ = ART / "trajectory"
TRAJ.mkdir(parents=True, exist_ok=True)

# Initial source/passive poses
T_cam_board = np.load(ROOT / "artifacts/calibration/extrinsics.npy")
Ta = np.load(PER / "source_active_pose_cam.npy")
Tp = np.load(PER / "source_passive_pose_cam.npy")

T_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Ta
T_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ Tp

scene = SceneBuilder().build(
    CONFIG["active"], CONFIG["passive"],
    T_active, T_passive, T_active,
    ROOT / "models/generated_xmls" / f"{NAME}_traj_test.xml",
)

grasps = GraspPlanner(ACTIVE, None, None).load_hardcoded()
data = mujoco.MjData(scene.model)
base_qpos = scene.data.qpos.copy()

active_joint = int(scene.model.body_jntadr[scene.active_body_id])
active_qadr = int(scene.model.jnt_qposadr[active_joint])
active_dadr = int(scene.model.jnt_dofadr[active_joint])

state = {
    "mode": "edit",
    "target": "goal",
    "source": T_active.copy(),
    "goal": T_active.copy(),
    "plan": False,
    "reset": False,
    "stop": False,
}

keys = deque()
planner = trajectory = None
stage_i = point_i = 0
stage_start = None
attached = False


def set_active(d, T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    d.qpos[active_qadr:active_qadr + 3] = T[:3, 3]
    d.qpos[active_qadr + 3:active_qadr + 7] = q[[3, 0, 1, 2]]
    d.qvel[active_dadr:active_dadr + 6] = 0


def set_goal(d, T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    d.mocap_pos[scene.goal_mocap_id] = T[:3, 3]
    d.mocap_quat[scene.goal_mocap_id] = q[[3, 0, 1, 2]]


def reset_view():
    data.qpos[:] = base_qpos
    data.qvel[:] = 0
    set_active(data, state["source"])
    set_goal(data, state["goal"])
    mujoco.mj_forward(scene.model, data)


def prepare_planner():
    scene.data.qpos[:] = base_qpos
    scene.data.qvel[:] = 0
    set_active(scene.data, state["source"])
    set_goal(scene.data, state["goal"])
    mujoco.mj_forward(scene.model, scene.data)


def move_target(dp=None, axis=None, deg=0):
    T = state[state["target"]].copy()

    if dp is not None:
        T[:3, 3] += np.asarray(dp)
    else:
        dR = Rotation.from_euler(axis, deg, degrees=True).as_matrix()
        T[:3, :3] = T[:3, :3] @ dR

    state[state["target"]] = T

    if state["target"] == "source":
        set_active(data, T)
    else:
        set_goal(data, T)


# Important: callback touches NO MuJoCo state.
def key_callback(key):
    keys.append(key)


def handle_key(key):
    if key == ord("R"):
        state["reset"] = True
        return

    if state["mode"] != "edit":
        return

    if key == ord("1"):
        state["target"] = "source"
        print("[test]: editing SOURCE")
    elif key == ord("2"):
        state["target"] = "goal"
        print("[test]: editing GOAL")
    elif key == 263: move_target([-POS_STEP, 0, 0])
    elif key == 262: move_target([ POS_STEP, 0, 0])
    elif key == 264: move_target([0, -POS_STEP, 0])
    elif key == 265: move_target([0,  POS_STEP, 0])
    elif key == 267: move_target([0, 0, -POS_STEP])
    elif key == 266: move_target([0, 0,  POS_STEP])
    elif key == ord("Q"): move_target(axis="x", deg= ROT_STEP)
    elif key == ord("A"): move_target(axis="x", deg=-ROT_STEP)
    elif key == ord("W"): move_target(axis="y", deg= ROT_STEP)
    elif key == ord("S"): move_target(axis="y", deg=-ROT_STEP)
    elif key == ord("E"): move_target(axis="z", deg= ROT_STEP)
    elif key == ord("D"): move_target(axis="z", deg=-ROT_STEP)
    elif key in (257, 335):
        state["plan"] = True


def attach_object():
    eef = data.body(EEF_BODY_NAME)
    T = np.eye(4)
    T[:3, 3] = eef.xpos
    T[:3, :3] = eef.xmat.reshape(3, 3)
    T = T @ np.linalg.inv(planner.chosen_grasp.T_object_eef)
    set_active(data, T)


old_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, lambda *_: state.__setitem__("stop", True))

try:
    with mujoco.viewer.launch_passive(
        scene.model, data, key_callback=key_callback
    ) as viewer:

        reset_view()

        print(
            "1=SOURCE | 2=GOAL | Arrows=XY | PgUp/PgDn=Z | "
            "Q/A W/S E/D=RPY | Enter=plan | R=reset/edit | Ctrl+C=exit"
        )
        print("[test]: editing GOAL")

        while viewer.is_running() and not state["stop"]:

            # Process UI events on main thread.
            while keys:
                handle_key(keys.popleft())

            # Reset robot + object to latest source; keep latest goal.
            if state["reset"]:
                state["reset"] = False
                state["mode"] = "edit"
                trajectory = None
                reset_view()
                print(f"[test]: reset; editing {state['target'].upper()}")

            # Plan latest source -> latest goal.
            if state["plan"]:
                state["plan"] = False
                reset_view()
                prepare_planner()

                planner = TrajectoryPlanner(scene)
                trajectory = planner.plan(
                    q_start=np.zeros(7),
                    T_mujoco_object_src=state["source"],
                    T_mujoco_object_goal=state["goal"],
                    grasps=grasps,
                    output_path=TRAJ / "test_trajectory.json",
                )

                if not trajectory:
                    state["mode"] = "edit"
                    reset_view()
                    print("[test]: planning failed; adjust poses and press Enter again.")
                else:
                    np.save(TRAJ / "test_source.npy", state["source"])
                    np.save(TRAJ / "test_goal.npy", state["goal"])

                    state["mode"] = "replay"
                    stage_i = point_i = 0
                    stage_start = None
                    attached = False
                    reset_view()
                    print("[test]: replaying continuously; R=reset/edit")

            # Replay.
            if state["mode"] == "replay" and trajectory:
                if stage_i >= len(trajectory):
                    reset_view()
                    stage_i = point_i = 0
                    stage_start = None
                    attached = False

                stage = trajectory[stage_i]

                if stage.type == "grasp":
                    data.qpos[planner.finger_qadrs] = 0.025
                    attached = True
                    stage_i += 1
                    stage_start = None

                elif stage.type == "release":
                    attached = False
                    data.qpos[planner.finger_qadrs] = 0.0
                    stage_i += 1
                    stage_start = None

                else:
                    if stage_start is None:
                        stage_start = time.monotonic()
                        point_i = 0

                    elapsed = time.monotonic() - stage_start

                    while (
                        point_i < len(stage.q_path)
                        and elapsed >= stage.time_from_start[point_i]
                    ):
                        data.qpos[planner.arm_qadrs] = stage.q_path[point_i]
                        mujoco.mj_forward(scene.model, data)

                        if attached:
                            attach_object()

                        point_i += 1

                    if point_i >= len(stage.q_path):
                        stage_i += 1
                        stage_start = None

            mujoco.mj_forward(scene.model, data)
            viewer.sync()
            time.sleep(0.001)

finally:
    signal.signal(signal.SIGINT, old_sigint)

print("[test]: exited cleanly.")