"""Interactive test for pipeline steps 6-11.

The MuJoCo teleoperation scene replaces steps 1-5. The poses recorded in the
viewer are treated as if FoundationPose returned them.

Usage:
    python test_steps_6_11.py mug_on_rack

Teleop keys:
    C        cycle controlled object: active -> passive -> goal_active
    Up/Down  +Y / -Y translation (world frame)
    Left/Right -X / +X translation (world frame)
    R / F    +Z / -Z translation (world frame)
    I / K    +roll / -roll  (local X)
    J / L    +pitch / -pitch (local Y)
    U / O    +yaw / -yaw (local Z)
    1        record SOURCE snapshot = current active + current passive
    2        record GOAL snapshot   = current goal_active + current passive
    Enter    finish teleop once both snapshots have been recorded
    H        print help
"""

import argparse
import queue
import time
from pathlib import Path

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from src.digital_twin import DigitalTwin, SceneBuilder
from src.render import render_for_vlm
from src.trajectory import GraspPlanner, TrajectoryPlanner
from src.vlm import get_sampling_constraints, get_validation_program, select_best_candidate


TRANSLATION_STEP = 0.01       # m per key press
ROTATION_STEP_DEG = 5.0       # deg per key press


def pause(stage):
    input(f"\n[test] Step {stage} complete. Press Enter to continue...")


def pose_from_body(data, body_id):
    T = np.eye(4)
    T[:3, 3] = data.xpos[body_id]
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return T


def set_free_body_pose(model, data, body_id, T):
    joint_id = int(model.body_jntadr[body_id])
    qadr = int(model.jnt_qposadr[joint_id])
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    data.qpos[qadr:qadr + 3] = T[:3, 3]
    data.qpos[qadr + 3:qadr + 7] = quat_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def set_fixed_body_pose(model, data, body_id, T):
    # The passive object is a direct child of worldbody in SceneBuilder, so
    # body_pos/body_quat are its world pose.
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    model.body_pos[body_id] = T[:3, 3]
    model.body_quat[body_id] = quat_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def pose_from_mocap(data, mocap_id):
    T = np.eye(4)
    q = data.mocap_quat[mocap_id]
    T[:3, 3] = data.mocap_pos[mocap_id]
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return T


def set_mocap_pose(model, data, mocap_id, T):
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    data.mocap_pos[mocap_id] = T[:3, 3]
    data.mocap_quat[mocap_id] = quat_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def print_help():
    print(
        "\nTeleop controls\n"
        "  C          cycle active / passive / goal_active\n"
        "  Up/Down    +Y / -Y\n"
        "  Left/Right -X / +X\n"
        "  R/F        +Z / -Z\n"
        "  I/K      +roll / -roll (local X)\n"
        "  J/L      +pitch / -pitch (local Y)\n"
        "  U/O      +yaw / -yaw (local Z)\n"
        "  1        record source active+passive snapshot\n"
        "  2        record goal goal_active+passive snapshot\n"
        "  Enter    continue after both snapshots are recorded\n"
        "  H        print this help\n"
    )


def teleop_and_record(scene):
    model = scene.model
    data = scene.data

    targets = ["active", "passive", "goal_active"]
    state = {
        "target_index": 0,
        "source": None,
        "goal": None,
        "done": False,
    }

    def get_pose(target):
        if target == "active":
            return pose_from_body(data, scene.active_body_id)
        if target == "passive":
            return pose_from_body(data, scene.passive_body_id)
        return pose_from_mocap(data, scene.goal_mocap_id)

    def set_pose(target, T):
        if target == "active":
            set_free_body_pose(model, data, scene.active_body_id, T)
        elif target == "passive":
            set_fixed_body_pose(model, data, scene.passive_body_id, T)
        else:
            set_mocap_pose(model, data, scene.goal_mocap_id, T)

    def nudge_translation(T, delta):
        out = T.copy()
        out[:3, 3] += np.asarray(delta, dtype=float)
        return out

    def nudge_rotation(T, axis, angle_deg):
        out = T.copy()
        R_delta = Rotation.from_euler(axis, angle_deg, degrees=True).as_matrix()
        out[:3, :3] = out[:3, :3] @ R_delta  # local-axis rotation
        return out

    key_queue = queue.SimpleQueue()

    # MuJoCo's viewer invokes this callback from the UI thread. Do not touch
    # mjModel/mjData here; queue the key and handle it in the main thread.
    def key_callback(keycode):
        key_queue.put(keycode)

    def handle_key(keycode):
        target = targets[state["target_index"]]

        if keycode == glfw.KEY_C:
            state["target_index"] = (state["target_index"] + 1) % len(targets)
            print(f"[teleop] controlling: {targets[state['target_index']]}")
            return

        if keycode == glfw.KEY_H:
            print_help()
            return

        if keycode == glfw.KEY_1:
            state["source"] = {
                "active": get_pose("active").copy(),
                "passive": get_pose("passive").copy(),
            }
            print("[teleop] recorded SOURCE = active + passive")
            return

        if keycode == glfw.KEY_2:
            state["goal"] = {
                "active": get_pose("goal_active").copy(),
                "passive": get_pose("passive").copy(),
            }
            print("[teleop] recorded GOAL = goal_active + passive")
            return

        if keycode in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            if state["source"] is None or state["goal"] is None:
                print("[teleop] record both SOURCE [1] and GOAL [2] first")
            else:
                state["done"] = True
            return

        T = get_pose(target)

        translations = {
            glfw.KEY_UP:    (0, +TRANSLATION_STEP, 0),
            glfw.KEY_DOWN:  (0, -TRANSLATION_STEP, 0),
            glfw.KEY_LEFT:  (-TRANSLATION_STEP, 0, 0),
            glfw.KEY_RIGHT: (+TRANSLATION_STEP, 0, 0),
            glfw.KEY_R:     (0, 0, +TRANSLATION_STEP),
            glfw.KEY_F:     (0, 0, -TRANSLATION_STEP),
        }
        rotations = {
            glfw.KEY_I: ("x", +ROTATION_STEP_DEG),
            glfw.KEY_K: ("x", -ROTATION_STEP_DEG),
            glfw.KEY_J: ("y", +ROTATION_STEP_DEG),
            glfw.KEY_L: ("y", -ROTATION_STEP_DEG),
            glfw.KEY_U: ("z", +ROTATION_STEP_DEG),
            glfw.KEY_O: ("z", -ROTATION_STEP_DEG),
        }

        if keycode in translations:
            set_pose(target, nudge_translation(T, translations[keycode]))
        elif keycode in rotations:
            axis, angle = rotations[keycode]
            set_pose(target, nudge_rotation(T, axis, angle))

    print_help()
    print("[teleop] controlling: active")

    # The key callback only queues input. All model/data access happens here
    # under the viewer lock, preventing mj_copyDataVisual races.
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not state["done"]:
            with viewer.lock():
                while True:
                    try:
                        keycode = key_queue.get_nowait()
                    except queue.Empty:
                        break
                    handle_key(keycode)

                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(0.01)

        if state["done"] and viewer.is_running():
            viewer.close()

    if state["source"] is None or state["goal"] is None:
        raise RuntimeError("Teleop window closed before both snapshots were recorded")

    return state["source"], state["goal"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    args = parser.parse_args()

    benchmark_name = args.benchmark
    config = BENCHMARK_CONFIGS[benchmark_name]
    active_catalog = OBJECT_CATALOG[config["active"]]
    passive_catalog = OBJECT_CATALOG[config["passive"]]

    root_dir = Path(__file__).resolve().parents[1]
    test_dir = root_dir / "artifacts" / benchmark_name / "test_main"
    render_dir = test_dir / "render"
    vlm_dir = test_dir / "vlm"
    dtwin_dir = test_dir / "dtwin"
    trajectory_dir = test_dir / "trajectory"
    xml_dir = root_dir / "models" / "generated_xmls"

    for d in (test_dir, render_dir, vlm_dir, dtwin_dir, trajectory_dir, xml_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Interactive replacement for steps 1-5.
    # These transforms are only starting points for the pose editor.
    # ------------------------------------------------------------------
    # The ArUco board origin is on the tabletop in this setup.
    table_surface_z = float(T_mujoco_board[2, 3])

    # Place each object's lower face on the tabletop, assuming the catalog
    # dimensions are centered about the object's local origin.
    T_active_init = np.eye(4)
    T_active_init[:3, 3] = [0.12, 0.00, table_surface_z]

    T_passive_init = np.eye(4)
    T_passive_init[:3, 3] = [-0.08, 0.00, table_surface_z]

    T_goal_init = np.eye(4)
    T_goal_init[:3, 3] = [0.00, 0.15, table_surface_z]

    scene_builder = SceneBuilder()
    teleop_scene = scene_builder.build(
        config["active"],
        config["passive"],
        T_active_init,
        T_passive_init,
        T_goal_init,
        output_path=xml_dir / f"{benchmark_name}_teleop.xml",
    )

    source_snapshot, goal_snapshot = teleop_and_record(teleop_scene)

    # Treat the snapshots exactly like FP detections. We choose a synthetic
    # camera frame such that the camera->MuJoCo conversion is the identity:
    # T_mujoco_board @ inv(T_cam_board) = I.
    T_cam_board = T_mujoco_board.copy()
    T_cam_active_source = source_snapshot["active"]
    T_cam_passive_source = source_snapshot["passive"]
    T_cam_active_edited = goal_snapshot["active"]
    T_cam_passive_edited = goal_snapshot["passive"]

    np.save(test_dir / "T_cam_active_source.npy", T_cam_active_source)
    np.save(test_dir / "T_cam_passive_source.npy", T_cam_passive_source)
    np.save(test_dir / "T_cam_active_edited.npy", T_cam_active_edited)
    np.save(test_dir / "T_cam_passive_edited.npy", T_cam_passive_edited)

    T_passive_active_edited = np.linalg.inv(T_cam_passive_edited) @ T_cam_active_edited
    T_mujoco_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_active_source
    T_mujoco_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_passive_source

    # Re-anchor the edited active/passive relation to the measured SOURCE
    # passive pose. This is the passive-frame stabilization used by the method.
    T_mujoco_active_goal = T_mujoco_passive @ T_passive_active_edited

    print("\n[test] synthetic FP snapshots recorded.")
    input("[test] Press Enter to begin step 6...")

    # ------------------------------------------------------------------
    # 6. Render poses
    # ------------------------------------------------------------------
    T_passive_world = np.linalg.inv(T_mujoco_passive)

    for i in range(2):
        view = config["render_config"][i]
        render_for_vlm(
            active_path=active_catalog["mesh"],
            passive_path=passive_catalog["mesh"],
            output_path=render_dir / f"translation_{i}.png",
            T_passive_active=T_passive_active_edited,
            T_passive_world=T_passive_world.copy(),
            rendered_frame="world",
            lookat=view["lookat"],
            front=view["front"],
            up=view["up"],
            zoom=view["zoom"],
        )

        render_for_vlm(
            active_path=active_catalog["mesh"],
            passive_path=passive_catalog["mesh"],
            output_path=render_dir / f"rotation_{i}.png",
            T_passive_active=T_passive_active_edited,
            rendered_frame="active",
            lookat=view["lookat"],
            front=view["front"],
            up=view["up"],
            zoom=view["zoom"],
        )

    pause(6)

    # ------------------------------------------------------------------
    # 7. VLM sampling constraints S + validation program V
    # ------------------------------------------------------------------
    S = get_sampling_constraints(
        instruction=config["instruction"],
        active_object=config["active"],
        passive_object=config["passive"],
        active_dimensions=active_catalog["dimensions"],
        translation_image_paths=[render_dir / "translation_0.png", render_dir / "translation_1.png"],
        rotation_image_paths=[render_dir / "rotation_0.png", render_dir / "rotation_1.png"],
        translation_output_path=vlm_dir / "translation_constraints.json",
        rotation_output_path=vlm_dir / "rotation_constraints.json",
    )
    V = get_validation_program(
        instruction=config["instruction"],
        active_object=config["active"],
        passive_object=config["passive"],
        active_dimensions=active_catalog["dimensions"],
        passive_dimensions=passive_catalog["dimensions"],
        input_image_paths=[render_dir / "translation_0.png", render_dir / "translation_1.png"],
        output_path=vlm_dir / "validation_program.json",
    )
    if V is None:
        raise RuntimeError("get_validation_program() returned None; make sure vlm.py returns V")

    print(f"[test] S = {S}")
    print(f"[test] V = {V}")
    pause(7)

    # ------------------------------------------------------------------
    # 8. Digital-twin sampling + render valid candidates
    # ------------------------------------------------------------------
    scene = scene_builder.build(
        config["active"],
        config["passive"],
        T_mujoco_active,
        T_mujoco_passive,
        T_mujoco_active_goal,
        output_path=xml_dir / f"{benchmark_name}_test_pipeline.xml",
    )

    digital_twin = DigitalTwin(scene, V, S)
    candidates = digital_twin.sample_and_validate(
        n_samples=1000,
        max_valid=5,
        sim_steps=500,
        visualize=True,
    )
    if not candidates:
        raise RuntimeError("No physically valid goal candidates found")

    np.save(dtwin_dir / "valid_candidates.npy", np.stack(candidates))

    candidate_image_paths = []
    for i, T_mujoco_active_goal_valid in enumerate(candidates):
        T_passive_active_valid = np.linalg.inv(T_mujoco_passive) @ T_mujoco_active_goal_valid
        image_path = render_dir / f"valid_candidate_{i}.png"

        render_for_vlm(
            active_path=active_catalog["mesh"],
            passive_path=passive_catalog["mesh"],
            output_path=image_path,
            T_passive_active=T_passive_active_valid,
            rendered_frame="active",
            lookat=config["render_config"][1]["lookat"],
            front=config["render_config"][1]["front"],
            up=config["render_config"][1]["up"],
            zoom=config["render_config"][1]["zoom"],
        )

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read candidate render: {image_path}")
        cv2.putText(image, f"candidate_{i}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(image_path), image)
        candidate_image_paths.append(image_path)

    pause(8)

    # ------------------------------------------------------------------
    # 9. VLM selects best valid candidate
    # ------------------------------------------------------------------
    best_index = select_best_candidate(
        config["instruction"],
        config["active"],
        config["passive"],
        input_image_paths=candidate_image_paths,
        output_path=vlm_dir / "best_candidate.json",
    )
    if not 0 <= best_index < len(candidates):
        raise RuntimeError(f"VLM returned invalid candidate index: {best_index}")

    T_mujoco_active_goal_selected = candidates[best_index]
    np.save(dtwin_dir / "selected_goal.npy", T_mujoco_active_goal_selected)
    print(f"[test] selected candidate: {best_index}")
    pause(9)

    # ------------------------------------------------------------------
    # 10. Grasp planning
    # ------------------------------------------------------------------
    # K is irrelevant when use_anygrasp=False, but GraspPlanner expects one.
    fake_K = np.array([
        [600.0, 0.0, 640.0],
        [0.0, 600.0, 360.0],
        [0.0, 0.0, 1.0],
    ])
    anygrasp_checkpoint = root_dir.parent / "anygrasp_sdk" / "grasp_detection" / "checkpoints" / "checkpoint_detection.tar"

    grasp_planner = GraspPlanner(active_catalog, fake_K, anygrasp_checkpoint)
    candidate_grasps = grasp_planner.plan(
        T_cam_object=T_cam_active_source,
        use_anygrasp=False,
    )
    if not candidate_grasps:
        raise RuntimeError("No candidate grasps found")

    print(f"[test] grasp candidates: {len(candidate_grasps)}")
    pause(10)

    # ------------------------------------------------------------------
    # 11. Trajectory planning
    # ------------------------------------------------------------------
    q_start = np.zeros(7)
    trajectory_planner = TrajectoryPlanner(scene)
    stages = trajectory_planner.plan(
        q_start=q_start,
        T_mujoco_object_src=T_mujoco_active,
        T_mujoco_object_goal=T_mujoco_active_goal_selected,
        grasps=candidate_grasps,
        output_path=trajectory_dir / "trajectory.json",
    )
    if stages is None:
        raise RuntimeError("No grasp produced a feasible trajectory")

    print(f"[test] trajectory saved to {trajectory_dir / 'trajectory.json'}")
    pause(11)

    print("[test] Replaying planned trajectory in MuJoCo...")
    trajectory_planner.replay_trajectory(stages)
    print("[test] Steps 6-11 test finished.")


if __name__ == "__main__":
    main()