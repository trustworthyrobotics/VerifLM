# scripts/test_dtwin.py

import queue, signal, time
from collections import Counter
from pathlib import Path

import os,sys
import glfw
import mujoco
import mujoco.viewer
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import DigitalTwin, SceneBuilder, SamplingConstraints, ValidationProgram


S = SamplingConstraints(
    x=.05, y=.05, z=.05, roll=15., pitch=15., yaw=15.,
    x_bias="none", y_bias="none", z_bias="none",
)

V = ValidationProgram(
    validation_mode="settling",
    required_contacts=["active_passive"],
    forbidden_contacts=["active_table"],
    xyz_relations=[],
    x_drift_tolerance=.0577, y_drift_tolerance=.0577, z_drift_tolerance=.0577,
    roll_drift_tolerance=60., pitch_drift_tolerance=60., yaw_drift_tolerance=60.
)

DP, DR = .01, 5.
SIM_STEPS, SAMPLE_BUDGET, MAX_VALID = 500, 100, 5
SYNC_EVERY, VIS_SLEEP = 5, .003

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


def mocap_pose(data, mid):
    T = np.eye(4)
    q = data.mocap_quat[mid]
    T[:3, 3] = data.mocap_pos[mid]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return T


def set_mocap(model, data, mid, T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    data.mocap_pos[mid] = T[:3, 3]
    data.mocap_quat[mid] = q[[3, 0, 1, 2]]
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
    active_catalog = OBJECT_CATALOG[cfg["active"]]
    passive_catalog = OBJECT_CATALOG[cfg["passive"]]
    z = float(T_mujoco_board[2, 3])

    def pose(x, y, z):
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        return T

    # Object origins are at their bottoms, so z = table surface.
    T_active = pose(.2, 0.2, z+active_catalog["dimensions"][2]/2)
    T_passive = pose(0.0, 0.2, z+passive_catalog["dimensions"][2]/2)
    T_goal = pose(0.0, 0.2, z+active_catalog["dimensions"][2]/2)

    out = Path("models/generated_xmls")
    out.mkdir(parents=True, exist_ok=True)

    scene = SceneBuilder().build(
        cfg["active"], cfg["passive"],
        T_active, T_passive, T_goal,
        output_path=out / "dtwin_test.xml",
    )

    dt = DigitalTwin(scene, V, S)
    model, data, mid = scene.model, scene.data, scene.goal_mocap_id

    keys = queue.SimpleQueue()
    stop = {"requested": False}
    valid_candidates = []
    candidate_idx = [-1]

    # Ctrl-C does not raise KeyboardInterrupt through MuJoCo.
    signal.signal(
        signal.SIGINT,
        lambda *_: stop.__setitem__("requested", True),
    )

    def sync(viewer, delay=0):
        if not stop["requested"] and viewer.is_running():
            viewer.sync()
            if delay:
                time.sleep(delay)

    def settle(T, viewer):
        """Show physical active at T, simulate settling, then validate."""
        if stop["requested"]:
            return False, T, "interrupted"

        with viewer.lock():
            dt.set_body_pose(data, dt.active_qadr, dt.active_dadr, T)
            valid, reason = dt.check_penetration(data, "initial")
            if valid:
                valid, reason = dt.check_forbidden_contact(data)

        sync(viewer)
        if not valid or stop["requested"]:
            return False, T, reason if not valid else "interrupted"

        if V.validation_mode == "settling":
            for i in range(SIM_STEPS):
                if stop["requested"]:
                    return False, T, "interrupted"

                with viewer.lock():
                    try:
                        mujoco.mj_step(model, data)
                        valid, reason = dt.check_forbidden_contact(data)
                    except mujoco.FatalError as e:
                        return False, T, f"mj_step failed: {e}"

                if not valid:
                    Tf = dt.get_body_pose(data, scene.active_body_id)
                    sync(viewer)
                    return False, Tf, reason

                if i % SYNC_EVERY == 0:
                    sync(viewer, VIS_SLEEP)

        with viewer.lock():
            Tf = dt.get_body_pose(data, scene.active_body_id)
            checks = [
                lambda: dt.check_penetration(data, "final"),
                lambda: dt.check_required_contact(data),
                lambda: dt.check_xyz_relations(data),
            ]
            if V.validation_mode == "settling":
                checks.append(lambda: dt.check_drift(T, Tf))

            for check in checks:
                valid, reason = check()
                if not valid:
                    return False, Tf, reason

        sync(viewer)
        return True, Tf, "valid"

    def validate(viewer):
        T = mocap_pose(data, mid)
        valid, Tf, reason = settle(T, viewer)
        if stop["requested"]:
            return

        print("\n=== VALIDATE ===")
        print("valid:", valid)
        print("reason:", reason)
        print("final pose:\n", np.round(Tf, 4))

    def sample(viewer):
        center = mocap_pose(data, mid)
        valid_poses, failures = [], Counter()
        print("\n=== SAMPLING ===")

        for i in range(SAMPLE_BUDGET):
            if stop["requested"]:
                break

            Ts = S.sample(center)

            with viewer.lock():
                set_mocap(model, data, mid, Ts)
            sync(viewer, .02)

            valid, Tf, reason = settle(Ts, viewer)
            if stop["requested"]:
                break

            if valid:
                valid_poses.append(Tf)
                print(f"[{i+1:03}] VALID ({len(valid_poses)}/{MAX_VALID})")
                if len(valid_poses) >= MAX_VALID:
                    break
            else:
                failures[reason] += 1
                print(f"[{i+1:03}] {reason}")

        if not stop["requested"]:
            valid_candidates[:] = valid_poses
            candidate_idx[0] = -1

            with viewer.lock():
                set_mocap(model, data, mid, center)
            sync(viewer)

            print(f"\nvalid candidates: {len(valid_poses)}")
            for reason, count in failures.most_common(5):
                print(f"{count:3}x  {reason}")
    def show_next_candidate(viewer):
        if not valid_candidates:
            print("[test] no valid candidates; press S first")
            return

        candidate_idx[0] = (candidate_idx[0] + 1) % len(valid_candidates)
        T = valid_candidates[candidate_idx[0]]

        with viewer.lock():
            set_mocap(model, data, mid, T)
            dt.set_body_pose(data, dt.active_qadr, dt.active_dadr, T)

        sync(viewer)
        print(f"[test] candidate {candidate_idx[0]+1}/{len(valid_candidates)}")
        print("""
Goal-active digital-twin test
  Arrows : XY
  R/F    : +/- Z
  I/K    : +/- roll
  J/L    : +/- pitch
  U/O    : +/- yaw
  V      : visibly validate + settle
  S      : visibly sample + settle
  Q      : quit
  Ctrl-C : quit cleanly
""")
    stop = {"requested": False}
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(
        signal.SIGINT,
        lambda *_: stop.__setitem__("requested", True),
    )

    try: 
        with mujoco.viewer.launch_passive(
            model, data, key_callback=lambda key: keys.put(key)
        ) as viewer:

            while viewer.is_running() and not stop["requested"]:
                try:
                    key = keys.get_nowait()
                except queue.Empty:
                    sync(viewer)
                    time.sleep(.01)
                    continue

                if key == glfw.KEY_Q:
                    stop["requested"] = True
                elif key == glfw.KEY_V:
                    validate(viewer)
                elif key == glfw.KEY_S:
                    sample(viewer)
                elif key == glfw.KEY_C:
                    show_next_candidate(viewer)
                elif key in TRANS or key in ROT:
                    with viewer.lock():
                        set_mocap(model, data, mid, nudge(mocap_pose(data, mid), key))
                    sync(viewer)

        print("[test] exited cleanly")
        sys.stdout.flush()
        os._exit(0)
    finally: 
        signal.signal(signal.SIGINT, old_handler)


if __name__ == "__main__":
    main()