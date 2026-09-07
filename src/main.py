import argparse
import copy
import json
import numpy as np
import cv2
from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS, T_mujoco_board
from pathlib import Path
from src.camera import Camera
from src.vlm import edit_image, estimate_physics, get_sampling_constraints, get_validation_program, load_sampling_constraints, load_validation_program, select_best_candidate
from src.perception import DepthEstimator, BBoxer, Segmenter, PoseEstimator
from src.render import render_for_vlm
from src.digital_twin import DigitalTwin, SceneBuilder
from src.trajectory import GraspPlanner, TrajectoryPlanner


# Read user args
parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
parser.add_argument("--reuse-extrinsics", action="store_true", default=False)
parser.add_argument("--reuse-source", action="store_true", default=False)
parser.add_argument("--reuse-edited", action="store_true", default=False)
parser.add_argument("--reuse-VS", action="store_true", default=False)
parser.add_argument("--reuse-perception", action="store_true", default=False)
parser.add_argument("--skip-candidate-selection", action="store_true", default=False)
parser.add_argument("--estimate-objects", action="store_true", help="use shared cached object estimates, reconstructing missing meshes and estimating mass/friction online")
parser.add_argument("--stage", type=int, default=0, help="stage number, zero-indexed")
args = parser.parse_args()

# Load benchmark config
BENCHMARK_NAME = args.benchmark
CONFIG = BENCHMARK_CONFIGS[BENCHMARK_NAME]
ACTIVE_NAME = CONFIG["active"][args.stage]
PASSIVE_NAME = CONFIG["passive"][args.stage]
INSTRUCTION = CONFIG["instruction"][args.stage]
RUNTIME_CATALOG = copy.deepcopy(OBJECT_CATALOG)
ACTIVE_CATALOG = RUNTIME_CATALOG[ACTIVE_NAME]
PASSIVE_CATALOG = RUNTIME_CATALOG[PASSIVE_NAME]

# Define directories
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT_DIR = ROOT_DIR / "artifacts" / f"{BENCHMARK_NAME}_stage{args.stage}"
CAMERA_DIR = OUTPUT_ROOT_DIR / "camera"
PERCEPTION_DIR = OUTPUT_ROOT_DIR / "perception"
RENDER_DIR = OUTPUT_ROOT_DIR / "render"
VLM_DIR = OUTPUT_ROOT_DIR / "vlm"
TRAJECTORY_DIR = OUTPUT_ROOT_DIR / "trajectory"
DTWIN_DIR = OUTPUT_ROOT_DIR / "dtwin"
OBJECT_ESTIMATION_DIR = ROOT_DIR / "artifacts" / "object_estimation"
XML_DIR = ROOT_DIR / "models" / "generated_xmls"
for dir in [OUTPUT_ROOT_DIR, CAMERA_DIR, PERCEPTION_DIR, RENDER_DIR, VLM_DIR, TRAJECTORY_DIR, DTWIN_DIR, OBJECT_ESTIMATION_DIR, XML_DIR]:
    dir.mkdir(exist_ok=True, parents=True)

MODELS_DIR = ROOT_DIR / "models"
GROUNDING_DINO_DIR = ROOT_DIR.parent / "GroundingDINO"
SAM2_DIR = ROOT_DIR.parent / "sam2"
FP_DIR = ROOT_DIR.parent / "FoundationPose"
ANYGRASP_DIR = ROOT_DIR.parent / "anygrasp_sdk" / "grasp_detection"
ANYGRASP_CHECKPOINT = ANYGRASP_DIR / "checkpoints" / "checkpoint_detection.tar"
EXTRINSICS_PATH = ROOT_DIR / "artifacts" / "calibration" / "extrinsics.npy"


def object_estimate_path(name):
    return OBJECT_ESTIMATION_DIR / name / "estimate.json"


def load_cached_object_estimate(name):
    path = object_estimate_path(name)
    if not path.exists():
        return None
    print(f"[main]: reusing shared object estimate for {name}: {path}")
    return json.loads(path.read_text())


# 1. Camera capture RGBD
# Output: rgbd observations stored in CAMERA_DIR
camera = Camera()
if args.reuse_source:
    rgb_path = CAMERA_DIR / "source_rgb.png"
    depth_path = CAMERA_DIR / "source_depth_m.npy"
    meta_path = CAMERA_DIR / "source_camera_meta.json"

    required = [rgb_path, depth_path, meta_path, EXTRINSICS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot reuse source; missing:\n" + "\n".join(missing)
        )

    metadata = json.loads(meta_path.read_text())
    camera.K = np.asarray(metadata["K"], dtype=np.float64)
    camera.dist_coeffs = np.asarray(metadata["dist_coeffs"], dtype=np.float64)
    T_cam_board = np.load(EXTRINSICS_PATH)
else:
    camera.start()
    try:
        camera.capture(
            render=True,
            save=True,
            reuse_extrinsics=args.reuse_extrinsics,
            extrinsics_path=EXTRINSICS_PATH,
            out_dir=CAMERA_DIR,
        )
    finally:
        camera.stop()

    T_cam_board = camera.get_extrinsics()

if T_cam_board is None:
    raise RuntimeError("Camera extrinsics unavailable")

input("step 1 done. press enter to proceed:")

# 2. Query VLM to edit the image
# Output: edited image stored in VLM_DIR
edited_path = VLM_DIR / "edited_rgb.png"
if args.reuse_edited:
    if not edited_path.exists():
        raise FileNotFoundError(f"Cannot reuse edited image; missing {edited_path}")
else:
    edit_image(instruction=INSTRUCTION, active_object=ACTIVE_NAME, passive_object=PASSIVE_NAME, input_image_path=CAMERA_DIR/"source_rgb.png", output_image_path=VLM_DIR/"edited_rgb.png")
input("step 2 done. press enter to proceed:")

# 3-5. Perception
pose_paths = {
    "source_active": PERCEPTION_DIR/"source_active_pose_cam.npy",
    "source_passive": PERCEPTION_DIR/"source_passive_pose_cam.npy",
    "edited_active": PERCEPTION_DIR/"edited_active_pose_cam.npy",
    "edited_passive": PERCEPTION_DIR/"edited_passive_pose_cam.npy",
}

if args.reuse_perception:
    required = [
        PERCEPTION_DIR/"source_vggt_depth_raw.npy",
        PERCEPTION_DIR/"edited_vggt_depth_raw.npy",
        PERCEPTION_DIR/"edited_vggt_depth_m.npy",
        PERCEPTION_DIR/"bboxes.json",
        PERCEPTION_DIR/"source_active_mask.png",
        PERCEPTION_DIR/"source_passive_mask.png",
        PERCEPTION_DIR/"edited_active_mask.png",
        PERCEPTION_DIR/"edited_passive_mask.png",
        *pose_paths.values(),
    ]
    if args.estimate_objects:
        required += [object_estimate_path(ACTIVE_NAME), object_estimate_path(PASSIVE_NAME)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot reuse perception; missing:\n" + "\n".join(missing))

    if args.estimate_objects:
        ACTIVE_CATALOG.update(json.loads(object_estimate_path(ACTIVE_NAME).read_text()))
        PASSIVE_CATALOG.update(json.loads(object_estimate_path(PASSIVE_NAME).read_text()))

    T_cam_active_source = np.load(pose_paths["source_active"])
    T_cam_passive_source = np.load(pose_paths["source_passive"])
    T_cam_active_edited = np.load(pose_paths["edited_active"])
    T_cam_passive_edited = np.load(pose_paths["edited_passive"])
    input("steps 3-5 done with cached result. press enter to proceed:")

else:
    # 3. VGGT depth estimation
    # Output: estimated depths stored in PERCEPTION_DIR
    depth_estimator = DepthEstimator()
    depth_estimator.predict(
        input_paths=[CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png"],
        output_paths=[PERCEPTION_DIR/"source_vggt_depth_raw.npy", PERCEPTION_DIR/"edited_vggt_depth_raw.npy"],
    )

    # 4. GroundingDINO + SAM2 + depth correction
    # Output: bboxes and masks stored in PERCEPTION_DIR
    bboxer = BBoxer(GROUNDING_DINO_DIR)
    bbox_list = bboxer.predict(
        input_paths=[CAMERA_DIR/"source_rgb.png", CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png", VLM_DIR/"edited_rgb.png"],
        prompts=[ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"], ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"]],
        output_path=PERCEPTION_DIR/"bboxes.json",
        loop=True,
    )

    segmenter = Segmenter(SAM2_DIR)
    segmenter.predict(
        input_paths=[CAMERA_DIR/"source_rgb.png", CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png", VLM_DIR/"edited_rgb.png"],
        bboxes=bbox_list,
        output_paths=[PERCEPTION_DIR/"source_active_mask.png", PERCEPTION_DIR/"source_passive_mask.png", PERCEPTION_DIR/"edited_active_mask.png", PERCEPTION_DIR/"edited_passive_mask.png"],
    )

    depth_estimator.correct(
        source_depth_paths=[CAMERA_DIR/"source_depth_m.npy", PERCEPTION_DIR/"source_vggt_depth_raw.npy"],
        source_mask_paths=[PERCEPTION_DIR/"source_active_mask.png", PERCEPTION_DIR/"source_passive_mask.png"],
        edited_depth_path=PERCEPTION_DIR/"edited_vggt_depth_raw.npy",
        output_path=PERCEPTION_DIR/"edited_vggt_depth_m.npy",
    )

    if args.estimate_objects:
        from src.object_estimation import ObjectEstimator
        estimator = ObjectEstimator(OBJECT_ESTIMATION_DIR)

        active_estimate = load_cached_object_estimate(ACTIVE_NAME)
        if active_estimate is None:
            active_estimate = estimator.estimate(
                ACTIVE_NAME, ACTIVE_CATALOG["prompt"], CAMERA_DIR/"source_rgb.png",
                CAMERA_DIR/"source_depth_m.npy", PERCEPTION_DIR/"source_active_mask.png", camera.get_intrinsics()
            )
        ACTIVE_CATALOG.update(active_estimate)

        passive_estimate = load_cached_object_estimate(PASSIVE_NAME)
        if passive_estimate is None:
            passive_estimate = estimator.estimate(
                PASSIVE_NAME, PASSIVE_CATALOG["prompt"], CAMERA_DIR/"source_rgb.png",
                CAMERA_DIR/"source_depth_m.npy", PERCEPTION_DIR/"source_passive_mask.png", camera.get_intrinsics()
            )
        PASSIVE_CATALOG.update(passive_estimate)

        physics = estimate_physics(
            INSTRUCTION, ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"],
            active_estimate["dimensions"], passive_estimate["dimensions"],
            CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png")

        active_estimate["physics"] = physics["active"]
        passive_estimate["physics"] = physics["passive"]

        object_estimate_path(ACTIVE_NAME).write_text(json.dumps(active_estimate, indent=2))
        object_estimate_path(PASSIVE_NAME).write_text(json.dumps(passive_estimate, indent=2))

        ACTIVE_CATALOG.update(active_estimate)
        PASSIVE_CATALOG.update(passive_estimate)

    input("steps 3-4 done. press enter to proceed:")

    # 5. FoundationPose pose estimation
    # Output: estimated object poses in camera frame stored in PERCEPTION_DIR
    pose_estimator = PoseEstimator(repo_path=FP_DIR)

    T_cam_active_source = pose_estimator.predict(
        rgb_path=CAMERA_DIR/"source_rgb.png",
        depth_path=CAMERA_DIR/"source_depth_m.npy",
        mask_path=PERCEPTION_DIR/"source_active_mask.png",
        mesh_path=ACTIVE_CATALOG["mesh"],
        K=camera.get_intrinsics(),
        output_dir=PERCEPTION_DIR,
        output_name="source_active_pose_cam.npy",
    )

    T_cam_passive_source = pose_estimator.predict(
        rgb_path=CAMERA_DIR/"source_rgb.png",
        depth_path=CAMERA_DIR/"source_depth_m.npy",
        mask_path=PERCEPTION_DIR/"source_passive_mask.png",
        mesh_path=PASSIVE_CATALOG["mesh"],
        K=camera.get_intrinsics(),
        output_dir=PERCEPTION_DIR,
        output_name="source_passive_pose_cam.npy",
    )

    T_cam_active_edited = pose_estimator.predict(
        rgb_path=VLM_DIR/"edited_rgb.png",
        depth_path=PERCEPTION_DIR/"edited_vggt_depth_m.npy",
        mask_path=PERCEPTION_DIR/"edited_active_mask.png",
        mesh_path=ACTIVE_CATALOG["mesh"],
        K=camera.get_intrinsics(),
        output_dir=PERCEPTION_DIR,
        output_name="edited_active_pose_cam.npy",
    )

    T_cam_passive_edited = pose_estimator.predict(
        rgb_path=VLM_DIR/"edited_rgb.png",
        depth_path=PERCEPTION_DIR/"edited_vggt_depth_m.npy",
        mask_path=PERCEPTION_DIR/"edited_passive_mask.png",
        mesh_path=PASSIVE_CATALOG["mesh"],
        K=camera.get_intrinsics(),
        output_dir=PERCEPTION_DIR,
        output_name="edited_passive_pose_cam.npy",
    )
    T_cam_passive_edited[:3,:3] = T_cam_passive_source[:3,:3]  # optionally keep passive object orientation unchanged, can comment out
    np.save(PERCEPTION_DIR/"edited_passive_pose_cam.npy", T_cam_passive_edited)

# Derive transformations for later use
T_passive_active_edited = np.linalg.inv(T_cam_passive_edited) @ T_cam_active_edited
T_mujoco_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_active_source
T_mujoco_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_passive_source
T_mujoco_active_goal = T_mujoco_passive @ T_passive_active_edited



# 6. Render poses
# Output: axis-labeled pictures of objects in RENDER_DIR
if not args.reuse_perception:
    # First pair of renderings: world-axis-labeled renders for translational reasoning
    render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"translation_0.png",
                T_passive_active=T_passive_active_edited, rendered_frame="passive",
                lookat=CONFIG["render_config"][0]["lookat"], front=CONFIG["render_config"][0]["front"], up=CONFIG["render_config"][0]["up"], zoom=CONFIG["render_config"][0]["zoom"])
    render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"translation_1.png",
                T_passive_active=T_passive_active_edited, rendered_frame="passive",
                lookat=CONFIG["render_config"][1]["lookat"], front=CONFIG["render_config"][1]["front"], up=CONFIG["render_config"][1]["up"], zoom=CONFIG["render_config"][1]["zoom"])

    # Second pair of renderings: active-object-axis-labeled renders for rotational reasoning
    render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"rotation_0.png",
                T_passive_active=T_passive_active_edited, rendered_frame="active",
                lookat=CONFIG["render_config"][0]["lookat"], front=CONFIG["render_config"][0]["front"], up=CONFIG["render_config"][0]["up"], zoom=CONFIG["render_config"][0]["zoom"])
    render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"rotation_1.png",
                T_passive_active=T_passive_active_edited, rendered_frame="active",
                lookat=CONFIG["render_config"][1]["lookat"], front=CONFIG["render_config"][1]["front"], up=CONFIG["render_config"][1]["up"], zoom=CONFIG["render_config"][1]["zoom"])
i = input("steps 5-6 done. press enter to proceed: ")


# 7. Query VLM for validation program V and sampling constraints S
# output: sampling constraints S and validation program V
if args.reuse_VS:
    S = load_sampling_constraints(VLM_DIR/"translation_constraints.json", VLM_DIR/"rotation_constraints.json", ACTIVE_CATALOG["dimensions"])
    V = load_validation_program(VLM_DIR/"validation_program.json", ACTIVE_CATALOG["dimensions"])
else:
    while True:
        S = get_sampling_constraints(instruction=INSTRUCTION, active_object=ACTIVE_NAME, passive_object=PASSIVE_NAME, 
                                    active_dimensions=ACTIVE_CATALOG["dimensions"], 
                                    translation_image_paths=[RENDER_DIR/"translation_0.png", RENDER_DIR/"translation_1.png"],
                                    rotation_image_paths=[RENDER_DIR/"rotation_0.png", RENDER_DIR/"rotation_1.png"],
                                    translation_output_path=VLM_DIR/"translation_constraints.json",
                                    rotation_output_path=VLM_DIR/"rotation_constraints.json")
        V = get_validation_program(instruction=INSTRUCTION, active_object=ACTIVE_NAME, passive_object=PASSIVE_NAME, 
                                active_dimensions=ACTIVE_CATALOG["dimensions"], passive_dimensions=PASSIVE_CATALOG["dimensions"],
                                input_image_paths=[RENDER_DIR/"translation_0.png", RENDER_DIR/"translation_1.png"],
                                output_path=VLM_DIR/"validation_program.json")
        i = input("Enter R to re-query VLM for V and S. Enter anything else to proceed to sampling in Mujoco: ")
        if i.lower() != "r":
            break

# 8. Sample in Mujoco + render successful candidates
scene_builder = SceneBuilder(RUNTIME_CATALOG)
obstacles = []
if args.stage > 0:
    # Iterate thru all previous stages and identify all active objects as possible obstacles
    # Technically, what should be done instead is identifying all objects mentioned in the config and
    # classify each one that's not relevant to the current task stage as obstacles.
    for stage in range(args.stage): 
        name = CONFIG["active"][stage]
        if name in (ACTIVE_NAME, PASSIVE_NAME) or any(name == n for n, _ in obstacles):
            continue
        prev_dir = ROOT_DIR / "artifacts" / f"{BENCHMARK_NAME}_stage{stage}"
        if args.estimate_objects:
            estimate_path = object_estimate_path(name)
            if estimate_path.exists():
                spec = dict(RUNTIME_CATALOG[name])
                spec.update(json.loads(estimate_path.read_text()))
                RUNTIME_CATALOG[name] = spec
        obstacles.append((name, np.load(prev_dir / "dtwin" / "T_mujoco_active_goal_selected.npy")))

scene = scene_builder.build(active_name=ACTIVE_NAME, passive_name=PASSIVE_NAME, 
                    T_mujoco_active=T_mujoco_active, T_mujoco_passive=T_mujoco_passive, T_mujoco_active_goal=T_mujoco_active_goal, obstacles=obstacles,
                    output_path=XML_DIR/f"{BENCHMARK_NAME}_stage{args.stage}.xml")
digital_twin = DigitalTwin(scene, V, S)
candidates = digital_twin.sample_and_validate(n_samples=1000, max_valid=5, sim_steps=500, visualize=True, video_path=DTWIN_DIR/f"sampling.mp4") # returns a list of valid T_mujoco_active_goal
if not candidates:
    raise RuntimeError("No physically valid goal candidates found.")

candidate_image_paths = []
for i, T_mujoco_active_goal_valid in enumerate(candidates):
    T_passive_active_valid = np.linalg.inv(T_mujoco_passive) @ T_mujoco_active_goal_valid
    render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/f"valid_candidate_{i}.png",
                   T_passive_active=T_passive_active_valid, rendered_frame="active",
                   lookat=CONFIG["render_config"][1]["lookat"], front=CONFIG["render_config"][1]["front"], up=CONFIG["render_config"][1]["up"], zoom=CONFIG["render_config"][1]["zoom"])

    array_path = DTWIN_DIR / f"valid_candidate_{i}.npy"
    image_path = RENDER_DIR / f"valid_candidate_{i}.png"
    image = cv2.imread(str(image_path))
    cv2.putText(image, f"candidate_{i}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(image_path), image)
    candidate_image_paths.append(image_path)
    np.save(array_path, T_mujoco_active_goal_valid)
input("step 8 done. press enter to proceed:")

# 9. Select best candidate
if args.skip_candidate_selection:
    best_index = input("Enter the index of the best candidate (0 to {}): ".format(len(candidates)-1))
    best_index = int(best_index)
else:
    best_index = select_best_candidate(INSTRUCTION, ACTIVE_NAME, PASSIVE_NAME, 
                                       input_image_paths=candidate_image_paths, output_path=VLM_DIR/"best_candidate.json")
T_mujoco_active_goal_selected = candidates[best_index]
np.save(DTWIN_DIR/"T_mujoco_active_goal_selected.npy", T_mujoco_active_goal_selected)
digital_twin.set_mocap_pose(scene.data, scene.refined_goal_mocap_id, T_mujoco_active_goal_selected)
input("step 9 done. press enter to proceed:")

# 10. Grasp Planning
grasp_planner = GraspPlanner(ACTIVE_CATALOG, camera.get_intrinsics(), ANYGRASP_CHECKPOINT)
candidate_grasps = grasp_planner.plan(T_cam_object=T_cam_active_source, 
                                      rgb_path=CAMERA_DIR/"source_rgb.png",
                                      depth_path=CAMERA_DIR/"source_depth_m.npy",
                                      mask_path=PERCEPTION_DIR/"source_active_mask.png",
                                      use_anygrasp=False)

# 11. Trajectory planning
q_start = np.zeros((7,))
trajectory_planner = TrajectoryPlanner(scene)
trajectory = trajectory_planner.plan(
    q_start=q_start,
    T_mujoco_object_src=T_mujoco_active,
    T_mujoco_object_goal=T_mujoco_active_goal_selected,
    grasps=candidate_grasps,
    output_path=TRAJECTORY_DIR/"trajectory.json",
)

if not trajectory:
    raise RuntimeError("No valid trajectory found.")
print(f"[main]: Trajectory saved to {TRAJECTORY_DIR}/trajectory.json")

trajectory_planner.replay_trajectory(trajectory)