from __future__ import annotations

import copy

from dotenv import load_dotenv

import pipeline.planner as planner
from pipeline.camera import capture_source
from pipeline.config import VLM_PROVIDERS
from pipeline.core import (
    PipelineArtifacts, StagePaths, save_pipeline_result, save_source, save_stage_poses,
)
from pipeline.debug import DebugController, ProgressLogger, wait_for_enter
from pipeline.digital_twin import GoalRefinementPipeline
from pipeline.perception import PerceptionPipeline
from pipeline.config import for_stage, pose_dict, stage_config
from pipeline.vlm import prepare_goal_image_vlm, resolve_object_prompts, resolve_task_stages


def run_pipeline(args) -> PipelineArtifacts:
    load_dotenv()
    stages = resolve_task_stages(args, provider=VLM_PROVIDERS.get("step_1", "openai"))
    object_poses = copy.deepcopy(args.object_poses or {})
    result = None

    for index, stage in enumerate(stages):
        stage_args = for_stage(args, stage, index, object_poses, len(stages))
        result = run_pipeline_single_stage(stage_args)
        object_poses[stage.active_object] = pose_dict(result.refined_goal_pose.T_world_object)
        save_stage_poses(args.benchmark, object_poses)
        if index + 1 < len(stages):
            wait_for_enter(f"Press Enter to continue to stage {index + 2}/{len(stages)}...")

    if result is None:
        raise RuntimeError("No task stage was executed.")
    return result


def run_pipeline_single_stage(args) -> PipelineArtifacts:
    paths = StagePaths.create(args.benchmark, stage_index=args.stage_index, stage_count=args.stage_count)
    config = stage_config(args)
    debug = DebugController(
        enabled=bool(getattr(args, "debug", False)),
        paths=paths,
        benchmark=args.benchmark,
        stage_index=args.stage_index,
        stage_count=args.stage_count,
    )
    progress = ProgressLogger(total=12, label=debug.label)

    progress.step(1, "loading/capturing source RGB-D")
    source = capture_source(args)
    save_source(paths, source)
    active_prompt, passive_prompt = resolve_object_prompts(args)

    progress.step(2, "loading or generating edited goal image")
    goal_path = paths.outputs / "goal_rgb.png"
    goal_bgr = prepare_goal_image_vlm(
        source.rgb_bgr,
        args.instruction,
        active_prompt,
        passive_prompt,
        goal_path,
        provider="gemini",
    )
    debug.show_goal_preview(source.rgb_bgr, goal_bgr)

    perception = PerceptionPipeline().run(
        source=source,
        goal_bgr=goal_bgr,
        active_prompt=active_prompt,
        passive_prompt=passive_prompt,
        active_mesh=args.active_mesh,
        passive_mesh=args.passive_mesh,
        paths=paths,
        debug=debug,
        progress=progress,
        camera_views=config.get("views"),
    )
    refinement = GoalRefinementPipeline().run(
        args=args,
        benchmark_config=config,
        perception=perception,
        active_prompt=active_prompt,
        passive_prompt=passive_prompt,
        paths=paths,
        debug=debug,
        progress=progress,
    )
    progress.step(12, "planning pick/place trajectory and saving outputs")
    planning = planner.plan_pick_place(
        args=args,
        xml_path=refinement.twin.result.xml_path,
        source_pose=refinement.twin.result.source_pose_mj,
        refined_goal_pose=refinement.refined_goal,
        goal_passive_pose=refinement.twin.result.passive_pose_mj,
        traj_dir=paths.trajectories,
    )

    refined_goal = refinement.refined_goal if planning is None else planning.refined_goal_pose
    trajectory = (
        {"status": "failed", "reason": "No collision-free grasp approach."}
        if planning is None
        else planning.trajectory
    )
    artifacts = PipelineArtifacts(
        source=source,
        goal=goal_bgr,
        source_pose=refinement.twin.result.source_pose_mj,
        raw_goal_pose=perception.goal_active,
        refined_goal_pose=refined_goal,
        trajectory=trajectory,
        metadata={
            "instruction": args.instruction,
            "active_object": active_prompt,
            "passive_object": passive_prompt,
            "relative_goal": perception.relative_goal.tolist(),
            "sampling_constraints": refinement.constraints.__dict__,
            "candidate_selection": refinement.selection,
        },
    )
    save_pipeline_result(paths, artifacts)
    return artifacts
