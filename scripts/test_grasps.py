#!/usr/bin/env python3

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
from src.trajectory import (
    GraspPlanner,
    TrajectoryPlanner,
    EEF_BODY_NAME,
)


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

XML_DIR = ROOT / "models" / "generated_xmls"
XML_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Initial object poses
# ---------------------------------------------------------------------

T_cam_board = np.load(ROOT / "artifacts/calibration/extrinsics.npy")
Ta = np.load(PER / "source_active_pose_cam.npy")
Tp = np.load(PER / "source_passive_pose_cam.npy")

T_active = (
    T_mujoco_board
    @ np.linalg.inv(T_cam_board)
    @ Ta
)

T_passive = (
    T_mujoco_board
    @ np.linalg.inv(T_cam_board)
    @ Tp
)


# ---------------------------------------------------------------------
# Build scene
#
# Goal pose is irrelevant here, so just place it at source.
# ---------------------------------------------------------------------

scene = SceneBuilder().build(
    CONFIG["active"],
    CONFIG["passive"],
    T_active,
    T_passive,
    T_active,
    XML_DIR / f"{NAME}_grasp_test.xml",
)

model = scene.model
data = mujoco.MjData(model)

base_qpos = scene.data.qpos.copy()


# ---------------------------------------------------------------------
# Active-object addresses
# ---------------------------------------------------------------------

active_joint = int(model.body_jntadr[scene.active_body_id])
active_qadr = int(model.jnt_qposadr[active_joint])
active_dadr = int(model.jnt_dofadr[active_joint])


# ---------------------------------------------------------------------
# Grasps + IK planner
# ---------------------------------------------------------------------

grasps = GraspPlanner(
    ACTIVE,
    None,
    None,
).load_hardcoded()

if not grasps:
    raise RuntimeError(
        f"{CONFIG['active']} has no hardcoded grasps."
    )

planner = TrajectoryPlanner(scene)


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

state = {
    "source": T_active.copy(),
    "grasp_i": None,
    "reset": False,
    "stop": False,
}

keys = deque()


# ---------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------

def set_active(d, T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()

    d.qpos[active_qadr:active_qadr + 3] = T[:3, 3]
    d.qpos[active_qadr + 3:active_qadr + 7] = q[[3, 0, 1, 2]]

    d.qvel[active_dadr:active_dadr + 6] = 0


def set_goal(d, T):
    """
    Keep transparent goal copy exactly on source so it does not distract.
    """
    q = Rotation.from_matrix(T[:3, :3]).as_quat()

    d.mocap_pos[scene.goal_mocap_id] = T[:3, 3]
    d.mocap_quat[scene.goal_mocap_id] = q[[3, 0, 1, 2]]


def reset_robot():
    """
    Reset robot to home while preserving current edited source pose.
    """
    data.qpos[:] = base_qpos
    data.qvel[:] = 0

    set_active(data, state["source"])
    set_goal(data, state["source"])

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------
# Source editing
# ---------------------------------------------------------------------

def move_source(dp=None, axis=None, deg=0):
    T = state["source"].copy()

    if dp is not None:
        # Translation in MuJoCo/world frame.
        T[:3, 3] += np.asarray(dp, dtype=float)

    else:
        # Rotation in active object's local frame.
        dR = Rotation.from_euler(
            axis,
            deg,
            degrees=True,
        ).as_matrix()

        T[:3, :3] = T[:3, :3] @ dR

    state["source"] = T

    # Once source changes, previous arm pose no longer corresponds
    # to the edited source. Return robot to home.
    state["grasp_i"] = None

    reset_robot()


# ---------------------------------------------------------------------
# Show one grasp
# ---------------------------------------------------------------------

def show_grasp(index):
    grasp = grasps[index]

    T_mujoco_eef = (
        state["source"]
        @ grasp.T_object_eef
    )

    # Make planner's internal scene agree with edited source.
    planner.ik_data.qpos[:] = base_qpos
    set_active(planner.ik_data, state["source"])
    mujoco.mj_forward(model, planner.ik_data)

    q = planner.solve_ik(T_mujoco_eef)

    print("\n" + "=" * 70)
    print(
        f"[test]: GRASP {index + 1}/{len(grasps)}"
    )
    print("=" * 70)

    print("source xyz:")
    print(state["source"][:3, 3])

    print("\nT_object_eef:")
    print(grasp.T_object_eef)

    print("\ntarget EEF xyz:")
    print(T_mujoco_eef[:3, 3])

    if q is None:
        print("\n[test]: IK FAILED")

        # Keep source visible, robot at home.
        reset_robot()
        return

    print("\n[test]: IK OK")
    print("q:")
    print(q)

    # Preserve current source object pose.
    data.qpos[:] = base_qpos
    data.qvel[:] = 0

    set_active(data, state["source"])
    set_goal(data, state["source"])

    # Place robot directly at grasp configuration.
    data.qpos[planner.arm_qadrs] = q

    # Close fingers for easier visual inspection.
    data.qpos[planner.finger_qadrs] = 0.025

    mujoco.mj_forward(model, data)

    # FK check.
    eef = data.body(EEF_BODY_NAME)

    print("\nactual EEF xyz:")
    print(eef.xpos)

    print(
        "EEF position error:",
        np.linalg.norm(
            eef.xpos - T_mujoco_eef[:3, 3]
        ),
    )


# ---------------------------------------------------------------------
# Keyboard
#
# IMPORTANT: callback only queues events.
# ---------------------------------------------------------------------

def key_callback(key):
    keys.append(key)


def handle_key(key):

    # Reset source to originally estimated pose.
    if key == ord("R"):
        state["source"] = T_active.copy()
        state["grasp_i"] = None

        reset_robot()

        print("[test]: source pose reset")
        return

    # -------------------------------------------------------------
    # Translation
    # -------------------------------------------------------------

    if key == 263:          # left
        move_source([-POS_STEP, 0, 0])

    elif key == 262:        # right
        move_source([POS_STEP, 0, 0])

    elif key == 264:        # down
        move_source([0, -POS_STEP, 0])

    elif key == 265:        # up
        move_source([0, POS_STEP, 0])

    elif key == 267:        # PgDn
        move_source([0, 0, -POS_STEP])

    elif key == 266:        # PgUp
        move_source([0, 0, POS_STEP])

    # -------------------------------------------------------------
    # Rotation
    # -------------------------------------------------------------

    elif key == ord("Q"):
        move_source(axis="x", deg=ROT_STEP)

    elif key == ord("A"):
        move_source(axis="x", deg=-ROT_STEP)

    elif key == ord("W"):
        move_source(axis="y", deg=ROT_STEP)

    elif key == ord("S"):
        move_source(axis="y", deg=-ROT_STEP)

    elif key == ord("E"):
        move_source(axis="z", deg=ROT_STEP)

    elif key == ord("D"):
        move_source(axis="z", deg=-ROT_STEP)

    # -------------------------------------------------------------
    # Grasp cycling
    # -------------------------------------------------------------

    elif key == ord("G"):
        if state["grasp_i"] is None:
            state["grasp_i"] = 0
        else:
            state["grasp_i"] = (
                state["grasp_i"] + 1
            ) % len(grasps)

        show_grasp(state["grasp_i"])

    elif key == ord("B"):
        if state["grasp_i"] is None:
            state["grasp_i"] = len(grasps) - 1
        else:
            state["grasp_i"] = (
                state["grasp_i"] - 1
            ) % len(grasps)

        show_grasp(state["grasp_i"])


# ---------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------

old_sigint = signal.getsignal(signal.SIGINT)

signal.signal(
    signal.SIGINT,
    lambda *_: state.__setitem__("stop", True),
)

try:
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        reset_robot()

        print(
            "\n"
            "SOURCE POSE EDITING\n"
            "-------------------\n"
            "Arrows      = X/Y translation\n"
            "PgUp/PgDn   = Z translation\n"
            "Q/A         = +/- roll\n"
            "W/S         = +/- pitch\n"
            "E/D         = +/- yaw\n"
            "\n"
            "GRASP VIEWING\n"
            "-------------\n"
            "G           = next hardcoded grasp\n"
            "B           = previous hardcoded grasp\n"
            "\n"
            "R           = reset source pose\n"
            "Ctrl+C      = exit\n"
        )

        print(
            f"[test]: loaded {len(grasps)} grasps "
            f"for {CONFIG['active']}"
        )

        while (
            viewer.is_running()
            and not state["stop"]
        ):

            while keys:
                handle_key(keys.popleft())

            viewer.sync()
            time.sleep(0.001)

finally:
    signal.signal(
        signal.SIGINT,
        old_sigint,
    )

print("[test]: exited cleanly.")