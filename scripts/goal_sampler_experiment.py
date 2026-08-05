from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import OBJECT_CATALOG, build_pipeline_config, stage_config
from pipeline.core import (
    T_from_pos_quat,
    ValidationProgram,
    mat_to_quat_wxyz,
    model_path,
    quat_wxyz_to_mat,
    rpy_to_quat,
)
from pipeline.digital_twin import (
    CandidateValidator,
    CandidateValidatorSettings,
    SamplerSettings,
    _body_transform,
    _goal_freejoint_qadr,
    _set_goal_pose,
    benchmark_xml_output_path,
    generate_benchmark_xml,
    render_relation,
    sample_and_validate_poses,
)


def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dtype=float,
    )


def _freejoint_qadr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Freejoint not found in MuJoCo model: {joint_name!r}")
    return int(model.jnt_qposadr[joint_id])


def _pose_from_freejoint(data: mujoco.MjData, qadr: int) -> tuple[np.ndarray, np.ndarray]:
    return data.qpos[qadr : qadr + 3].copy(), data.qpos[qadr + 3 : qadr + 7].copy()


def _print_controls(position_step: float, rotation_step_deg: float) -> None:
    print("")
    print("[controls]")
    print(f"  Arrow Left/Right : move X by {position_step:.3f} m")
    print(f"  Arrow Down/Up    : move Y by {position_step:.3f} m")
    print(f"  PageDown/PageUp  : move Z by {position_step:.3f} m")
    print(f"  Q/E              : roll by {rotation_step_deg:.1f} deg")
    print(f"  A/D              : pitch by {rotation_step_deg:.1f} deg")
    print(f"  Z/C              : yaw by {rotation_step_deg:.1f} deg")
    print("  Enter            : start sampling, or continue after a valid candidate")
    print("  Close viewer     : exit")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactively choose a goal-pose sampling mean, then run the MuJoCo sampler."
    )
    parser.add_argument("--benchmark", default="plates_on_rack")
    parser.add_argument("--camera-type", default="gemini")
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--position-step", type=float, default=0.005)
    parser.add_argument("--rotation-step-deg", type=float, default=2.0)
    parser.add_argument("--success-limit", type=int, default=10)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--render-candidates",
        action="store_true",
        help="Render valid candidate PNGs with render_relation. Disabled by default.",
    )
    parser.add_argument("--render-frame", choices=["active", "world"], default="active")
    args = parser.parse_args()

    from mujoco.glfw import glfw
    import mujoco.viewer

    runtime_args = build_pipeline_config(
        args.benchmark,
        camera_type=args.camera_type,
        axis_reasoning=False,
        debug=True,
    )
    if args.stage_index:
        from pipeline.config import for_stage
        from pipeline.core import TaskStage

        if not 0 <= args.stage_index < len(runtime_args.default_stages):
            raise IndexError(
                f"--stage-index {args.stage_index} is outside "
                f"{len(runtime_args.default_stages)} configured stage(s)."
            )
        selected = runtime_args.default_stages[args.stage_index]
        stage = TaskStage(
            selected["active"],
            selected["passive"],
            selected.get("reason", ""),
            selected.get("instruction", ""),
        )
        runtime_args = for_stage(
            runtime_args,
            stage,
            args.stage_index,
            dict(runtime_args.object_poses or {}),
            len(runtime_args.default_stages),
        )

    config = stage_config(runtime_args)
    xml_path = model_path(
        generate_benchmark_xml(
            runtime_args.benchmark,
            config,
            output_path=benchmark_xml_output_path(
                runtime_args.benchmark,
                stage_index=runtime_args.stage_index,
                stage_count=runtime_args.stage_count,
            ),
        )
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sampler_settings = SamplerSettings.from_config(config)
    contact_entities = config["scene"]["contact_entities"]
    validator = CandidateValidator(
        model,
        data,
        active_body_id=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            runtime_args.mujoco_active_body,
        ),
        passive_body_id=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            contact_entities["passive"],
        ),
        table_geom_ids=(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact_entities.get("table", "table"),
            ),
        ),
        active_dimensions=OBJECT_CATALOG[
            config["scene"]["active_object"]
        ]["dimensions"],
        validation_program=ValidationProgram.model_validate(
            config["digital_twin"]["validation_program"]
        ),
        settings=CandidateValidatorSettings.from_sampling(
            config["digital_twin"]["sampling"]
        ),
    )

    goal_qadr = _goal_freejoint_qadr(model, config["scene"].get("goal_freejoint_name"))
    active_qadr = _freejoint_qadr(model, runtime_args.object_freejoint_name)
    goal_pos, goal_quat = _pose_from_freejoint(data, active_qadr)
    _set_goal_pose(data, goal_qadr, goal_pos, goal_quat)
    mujoco.mj_forward(model, data)

    goal_rpy = _quat_to_rpy(goal_quat)
    position_step = float(args.position_step)
    rotation_step = np.deg2rad(float(args.rotation_step_deg))
    start_sampling = [False]
    valid_continue = [False]
    sample_pose_callbacks = [0]

    def apply_goal_pose() -> None:
        goal_quat_now = mat_to_quat_wxyz(quat_wxyz_to_mat(rpy_to_quat(*goal_rpy)))
        _set_goal_pose(data, goal_qadr, goal_pos, goal_quat_now)
        mujoco.mj_forward(model, data)

    def print_pose() -> None:
        rpy_deg = np.rad2deg(goal_rpy)
        print(
            "[mean] "
            f"xyz=({goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}) "
            f"rpy_deg=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f})",
            flush=True,
        )

    def key_callback(keycode: int) -> None:
        if keycode in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            start_sampling[0] = True
            valid_continue[0] = True
            return

        if start_sampling[0]:
            return

        changed = True
        if keycode == glfw.KEY_LEFT:
            goal_pos[0] -= position_step
        elif keycode == glfw.KEY_RIGHT:
            goal_pos[0] += position_step
        elif keycode == glfw.KEY_DOWN:
            goal_pos[1] -= position_step
        elif keycode == glfw.KEY_UP:
            goal_pos[1] += position_step
        elif keycode == glfw.KEY_PAGE_DOWN:
            goal_pos[2] -= position_step
        elif keycode == glfw.KEY_PAGE_UP:
            goal_pos[2] += position_step
        elif keycode == glfw.KEY_Q:
            goal_rpy[0] -= rotation_step
        elif keycode == glfw.KEY_E:
            goal_rpy[0] += rotation_step
        elif keycode == glfw.KEY_A:
            goal_rpy[1] -= rotation_step
        elif keycode == glfw.KEY_D:
            goal_rpy[1] += rotation_step
        elif keycode == glfw.KEY_Z:
            goal_rpy[2] -= rotation_step
        elif keycode == glfw.KEY_C:
            goal_rpy[2] += rotation_step
        else:
            changed = False

        if changed:
            apply_goal_pose()
            print_pose()

    def show_sample_pose(pos, quat) -> None:
        sample_pose_callbacks[0] += 1
        _set_goal_pose(data, goal_qadr, pos, quat)
        mujoco.mj_forward(model, data)
        viewer.sync()

    output_dir = None
    T_world_passive = None
    camera = None
    if args.render_candidates:
        output_dir = (
            Path(args.out_dir)
            if args.out_dir is not None
            else REPO_ROOT / "artifacts" / runtime_args.benchmark / "sampler_experiment"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("candidate_*.png"):
            stale.unlink()

        passive_name = config["scene"]["contact_entities"]["passive"]
        T_world_passive = _body_transform(model, data, passive_name)
        camera = tuple(config["views"])[0]

    def render_valid_candidate(candidate_number: int, info: dict) -> None:
        if not args.render_candidates:
            return
        assert output_dir is not None
        assert T_world_passive is not None
        assert camera is not None
        final_pos = info.get("final_pos")
        final_quat = info.get("final_quat")
        if final_pos is None or final_quat is None:
            print(f"[render:skip] valid candidate {candidate_number} has no final pose", flush=True)
            return
        image_path = output_dir / f"candidate_{candidate_number - 1:03d}.png"
        print(f"[render] writing {image_path}", flush=True)
        render_relation(
            active_mesh=model_path(runtime_args.active_mesh),
            passive_mesh=model_path(runtime_args.passive_mesh),
            T_world_active=T_from_pos_quat(final_pos, final_quat),
            T_world_passive=T_world_passive,
            out_path=image_path,
            camera=camera,
            frame=args.render_frame,
        )
        info["image_path"] = str(image_path)
        print(f"[render] wrote {image_path}", flush=True)

    def pause_on_valid(candidate_number: int, info=None) -> None:
        valid_continue[0] = False
        generated_count = sample_pose_callbacks[0] - candidate_number
        print(
            f"[valid {candidate_number}/{args.success_limit}] "
            f"total candidates generated so far: {generated_count}",
            flush=True,
        )
        if info is not None:
            render_valid_candidate(candidate_number, info)
        while not valid_continue[0] and viewer.is_running():
            viewer.sync()
            time.sleep(0.016)

    _print_controls(position_step, args.rotation_step_deg)
    print(f"[scene] loaded {xml_path}")
    print_pose()
    print("[experiment] adjust the green goal marker, then press Enter to sample")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while not start_sampling[0] and viewer.is_running():
            viewer.sync()
            time.sleep(0.016)
        if not viewer.is_running():
            return 0

        mean_pos = goal_pos.copy()
        mean_quat = data.qpos[goal_qadr + 3 : goal_qadr + 7].copy()
        sample_count = runtime_args.n_samples if args.sample_limit is None else int(args.sample_limit)
        print(
            f"[sampler] starting from chosen mean; "
            f"sample_limit={sample_count}, success_limit={args.success_limit}",
            flush=True,
        )
        candidates = sample_and_validate_poses(
            mean_pos,
            mean_quat,
            validator,
            sampler_settings,
            n_samples=sample_count,
            max_valid_candidates=args.success_limit,
            verbose=True,
            sample_callback=show_sample_pose,
            pause_on_valid=True,
            pause_callback=pause_on_valid,
        )

    if args.render_candidates:
        print(f"[done] rendered {len(candidates)} valid candidate(s) to {output_dir}")
    else:
        print(f"[done] found {len(candidates)} valid candidate(s); rendering disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
