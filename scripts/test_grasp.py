# scripts/test_grasps.py

import argparse, os, queue, signal, sys, time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import SceneBuilder
from src.trajectory import GraspPlanner, TrajectoryPlanner, ARM_JOINT_NAMES


DP, DR = .01, 5.
TRANS = {
    glfw.KEY_UP: (0, DP, 0), glfw.KEY_DOWN: (0, -DP, 0),
    glfw.KEY_LEFT: (-DP, 0, 0), glfw.KEY_RIGHT: (DP, 0, 0),
    glfw.KEY_R: (0, 0, DP), glfw.KEY_F: (0, 0, -DP),
}
ROT = {
    glfw.KEY_I: ("x", DR), glfw.KEY_K: ("x", -DR),
    glfw.KEY_J: ("y", DR), glfw.KEY_L: ("y", -DR),
    glfw.KEY_U: ("z", DR), glfw.KEY_O: ("z", -DR),
}


def pose(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def body_pose(data, bid):
    T = np.eye(4)
    T[:3, 3] = data.xpos[bid]
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    return T


def set_free_body(model, data, qadr, T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    data.qpos[qadr:qadr+7] = [*T[:3, 3], q[3], q[0], q[1], q[2]]
    mujoco.mj_forward(model, data)


def nudge(T, key):
    T = T.copy()
    if key in TRANS:
        T[:3, 3] += TRANS[key]
    else:
        axis, angle = ROT[key]
        T[:3, :3] @= Rotation.from_euler(axis, angle, degrees=True).as_matrix()
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    args = parser.parse_args()

    cfg = BENCHMARK_CONFIGS[args.benchmark]
    active = OBJECT_CATALOG[cfg["active"]]
    passive = OBJECT_CATALOG[cfg["passive"]]

    grasps = GraspPlanner(active, np.eye(3), None).load_hardcoded()
    if not grasps:
        raise RuntimeError(f"{cfg['active']} has no hardcoded grasps")

    table_z = float(T_mujoco_board[2, 3])
    T_active = pose(.2, .2, table_z + active["dimensions"][2]/2)
    T_passive = pose(0, .2, table_z + passive["dimensions"][2]/2)

    out = Path("models/generated_xmls")
    out.mkdir(parents=True, exist_ok=True)
    scene = SceneBuilder().build(
        cfg["active"], cfg["passive"],
        T_active, T_passive, T_active,
        output_path=out/"grasp_test.xml",
    )

    planner = TrajectoryPlanner(scene)
    model, data = scene.model, scene.data
    arm_qadrs = np.array([model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES])
    active_qadr = planner.collision_checker.active_qadr
    q_home = np.zeros(7)

    keys, stop, idx = queue.SimpleQueue(), {"requested": False}, [-1]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("requested", True))

    def set_arm(q):
        data.qpos[arm_qadrs] = q
        mujoco.mj_forward(model, data)

    def replay(path, viewer, dt=.015):
        for q in path:
            if stop["requested"] or not viewer.is_running():
                return False
            set_arm(q)
            viewer.sync()
            time.sleep(dt)
        return True

    def show_grasp(viewer):
        idx[0] = (idx[0] + 1) % len(grasps)
        grasp = grasps[idx[0]]
        T_obj = body_pose(data, scene.active_body_id)

        T_pregrasp = T_obj @ grasp.T_object_pregrasp_eef
        T_grasp = T_obj @ grasp.T_object_eef

        q_pregrasp = planner.solve_ik(T_pregrasp, q_home)
        q_grasp = planner.solve_ik(T_grasp, q_pregrasp) if q_pregrasp is not None else None

        if q_pregrasp is None or q_grasp is None:
            print(f"[{idx[0]+1}/{len(grasps)}] IK failed")
            set_arm(q_home)
            return

        path_home_pregrasp = planner.plan_trajectory(q_home, q_pregrasp)
        if not path_home_pregrasp:
            print(f"[{idx[0]+1}/{len(grasps)}] home -> pregrasp failed")
            set_arm(q_home)
            return

        path_pregrasp_grasp = planner.plan_grasp_approach(q_pregrasp, q_grasp)

        print(f"[{idx[0]+1}/{len(grasps)}] replaying home -> pregrasp -> grasp")
        set_arm(q_home)
        if replay(path_home_pregrasp, viewer):
            replay(path_pregrasp_grasp, viewer)

    def move_active(key):
        set_free_body(
            model, data, active_qadr,
            nudge(body_pose(data, scene.active_body_id), key),
        )
        set_arm(q_home)

    print("""
Hardcoded grasp test
  Enter  : next grasp + replay home -> pregrasp -> grasp
  Arrows : XY
  R/F    : +/- Z
  I/K    : +/- roll
  J/L    : +/- pitch
  U/O    : +/- yaw
  Q      : quit
""")

    try:
        set_arm(q_home)

        with mujoco.viewer.launch_passive(
            model, data, key_callback=lambda key: keys.put(key)
        ) as viewer:
            while viewer.is_running() and not stop["requested"]:
                try:
                    key = keys.get_nowait()
                except queue.Empty:
                    viewer.sync()
                    time.sleep(.01)
                    continue

                if key == glfw.KEY_Q:
                    stop["requested"] = True
                elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                    show_grasp(viewer)
                elif key in TRANS or key in ROT:
                    move_active(key)

                viewer.sync()

    finally:
        print("[test] exited cleanly")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()