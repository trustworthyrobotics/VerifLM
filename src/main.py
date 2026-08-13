import argparse
import numpy as np
from src.config import OBJECT_CATALOG, BENCHMARK_CONFIGS
from pathlib import Path
from src.camera import Camera
from src.vlm import edit_image, get_sampling_constraints, get_validation_program
from src.perception import DepthEstimator, BBoxer, Segmenter, PoseEstimator
from src.render import render_for_vlm
from src.digital_twin import T_mujoco_board, DigitalTwin, SceneBuilder, Scene, ValidationProgram, SamplingConstraints
from src.trajectory import GraspPlanner, MotionPlanner, TrajectoryPlanner

# Read user args
parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
parser.add_argument("--calc-extrinsics", action="store_true", default=False)
args = parser.parse_args()

# Load benchmark config
BENCHMARK_NAME = args.benchmark
CONFIG = BENCHMARK_CONFIGS[BENCHMARK_NAME]
ACTIVE_CATALOG = OBJECT_CATALOG[CONFIG["active"]]
PASSIVE_CATALOG = OBJECT_CATALOG[CONFIG["passive"]]

# Define directories
ROOT_DIR = Path(__file__).resolve()
OUTPUT_ROOT_DIR = Path("artifacts" / BENCHMARK_NAME)
CAMERA_DIR = OUTPUT_ROOT_DIR / "camera"
PERCEPTION_DIR = OUTPUT_ROOT_DIR / "perception"
RENDER_DIR = OUTPUT_ROOT_DIR / "render"
VLM_DIR = OUTPUT_ROOT_DIR / "vlm"
TRAJECTORY_DIR = OUTPUT_ROOT_DIR / "trajectory"
DTWIN_DIR = OUTPUT_ROOT_DIR / "dtwin"
XML_DIR = ROOT_DIR / "models" / "generated_xmls"
for dir in [OUTPUT_ROOT_DIR, CAMERA_DIR, PERCEPTION_DIR, RENDER_DIR, VLM_DIR, TRAJECTORY_DIR, DTWIN_DIR, XML_DIR]:
    dir.mkdir(exist_ok=True, parents=True)

GROUNDING_DINO_DIR = ROOT_DIR.parent / "GroundingDINO"
SAM2_DIR = ROOT_DIR.parent / "sam2"
FP_DIR = ROOT_DIR.parent / "FoundationPose"
ANYGRASP_DIR = ROOT_DIR.parent / "anygrasp_sdk" / "grasp_detection"
ANYGRASP_CHECKPOINT = ANYGRASP_DIR / "checkpoints" / "checkpoint_detection.tar"

# 1. Camera capture RGBD
# Output: rgbd observations stored in CAMERA_DIR
camera = Camera()
camera.start()
camera.capture(render=True, save=True, calculate_extrinsics=args.calc_extrinsics, out_dir=CAMERA_DIR)
camera.stop()

# 2. Query VLM to edit the image
# Output: edited image stored in VLM_DIR
edit_image(instruction=CONFIG["instruction"], active_object=CONFIG["active"], passive_object=CONFIG["passive"], input_image_path=CAMERA_DIR/"source_rgb.png", output_image_path=VLM_DIR/"edited_rgb.png")

# 3. VGGT depth estimation
# Output: estimated depths stored in PERCEPTION_DIR
depth_estimator = DepthEstimator()
depth_estimator.predict(input_paths=[CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png"], output_paths=[PERCEPTION_DIR/"source_vggt_depth_raw.npy", PERCEPTION_DIR/"edited_vggt_depth_raw.npy"])

# 4. GroundingDINO + SA2 loop (user provides bbox if necessary) + depth correction
# Output: bboxes and masks stored in PERCEPTION_DIR
bboxer = BBoxer(GROUNDING_DINO_DIR)
bbox_list = bboxer.predict(input_paths=[CAMERA_DIR/"source_rgb.png", CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png", VLM_DIR/"edited_rgb.png"], 
                            prompts=[ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"], ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"]],
                            output_path=PERCEPTION_DIR/"bboxes.json")
segmenter = Segmenter(SAM2_DIR)
segmenter.predict(input_paths=[CAMERA_DIR/"source_rgb.png", CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png", VLM_DIR/"edited_rgb.png"],
                  bboxes=bbox_list, 
                  output_paths=[PERCEPTION_DIR/"source_active_mask.png", PERCEPTION_DIR/"source_passive_mask.png", PERCEPTION_DIR/"edited_active_mask.png", PERCEPTION_DIR/"edited_passive_mask.png"])
depth_estimator.correct(source_depth_paths=[CAMERA_DIR/"source_depth_m.npy", PERCEPTION_DIR/"source_vggt_depth_raw.npy"], 
                        source_mask_paths=[PERCEPTION_DIR/"source_active_mask.png", PERCEPTION_DIR/"source_passive_mask.png"],
                        edited_depth_path=PERCEPTION_DIR/"edited_vggt_depth_raw.npy",
                        output_path=PERCEPTION_DIR/"edited_vggt_depth_m.npy")

# 5. FP pose estimation 
# Output: estimated object poses in world frame in PERCEPTION_DIR
pose_estimator = PoseEstimator(repo_path=FP_DIR)
T_cam_active_source = pose_estimator.predict(rgb_path=CAMERA_DIR/"source_rgb.png", depth_path=CAMERA_DIR/"source_depth_m.npy", 
                                            mask_path=PERCEPTION_DIR/"source_active_mask.png", mesh_path=ACTIVE_CATALOG["mesh"],
                                            K=camera.get_intrinsics(), output_dir=PERCEPTION_DIR, output_name="source_active_pose_cam.npy")
T_cam_passive_source = pose_estimator.predict(rgb_path=CAMERA_DIR/"source_rgb.png", depth_path=CAMERA_DIR/"source_depth_m.npy", 
                                            mask_path=PERCEPTION_DIR/"source_passive_mask.png", mesh_path=PASSIVE_CATALOG["mesh"],
                                            K=camera.get_intrinsics(), output_dir=PERCEPTION_DIR, output_name="source_passive_pose_cam.npy")

T_cam_active_edited = pose_estimator.predict(rgb_path=VLM_DIR/"edited_rgb.png", depth_path=PERCEPTION_DIR/"edited_vggt_depth_m.npy", 
                                            mask_path=PERCEPTION_DIR/"edited_active_mask.png", mesh_path=ACTIVE_CATALOG["mesh"],
                                            K=camera.get_intrinsics(), output_dir=PERCEPTION_DIR, output_name="edited_active_pose_cam.npy")
T_cam_passive_edited = pose_estimator.predict(rgb_path=VLM_DIR/"edited_rgb.png", depth_path=PERCEPTION_DIR/"edited_vggt_depth_m.npy", 
                                            mask_path=PERCEPTION_DIR/"edited_passive_mask.png", mesh_path=PASSIVE_CATALOG["mesh"],
                                            K=camera.get_intrinsics(), output_dir=PERCEPTION_DIR, output_name="edited_passive_pose_cam.npy")

# 6. Render poses
# Output: axis-labeled pictures of objects in RENDER_DIR

# First pair of renderings: world-axis-labeled renders for translational reasoning
T_cam_board = camera.get_extrinsics()
T_passive_board_edited = np.linalg.inv(T_cam_passive_edited) @ T_cam_board
render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"translation_0.png",
               T_passive_board=T_passive_board_edited, rendered_frame="world",
               lookat=CONFIG["render_config"][0]["lookat"], front=CONFIG["render_config"][0]["front"], up=CONFIG["render_config"][0]["up"], zoom=CONFIG["render_config"][0]["zoom"])
render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"translation_1.png",
               T_passive_board=T_passive_board_edited, rendered_frame="world",
               lookat=CONFIG["render_config"][1]["lookat"], front=CONFIG["render_config"][1]["front"], up=CONFIG["render_config"][1]["up"], zoom=CONFIG["render_config"][1]["zoom"])

# Second pair of renderings: active-object-axis-labeled renders for rotational reasoning
T_passive_active_edited = np.linalg.inv(T_cam_passive_edited) @ T_cam_active_edited
render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"rotation_0.png",
               T_passive_board=T_passive_active_edited, rendered_frame="active",
               lookat=CONFIG["render_config"][0]["lookat"], front=CONFIG["render_config"][0]["front"], up=CONFIG["render_config"][0]["up"], zoom=CONFIG["render_config"][0]["zoom"])
render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/"rotation_1.png",
               T_passive_board=T_passive_active_edited, rendered_frame="active",
               lookat=CONFIG["render_config"][1]["lookat"], front=CONFIG["render_config"][1]["front"], up=CONFIG["render_config"][1]["up"], zoom=CONFIG["render_config"][1]["zoom"])


# 7. Query VLM for validation program V and sampling constraints S
# output: sampling constraints S and validation program V
S = get_sampling_constraints(instruction=CONFIG["instruction"], active_object=CONFIG["active"], passive_object=CONFIG["passive"], 
                             active_dimensions=ACTIVE_CATALOG["dimensions"], 
                             translation_image_paths=[RENDER_DIR/"translation_0.png", RENDER_DIR/"translation_1.png"],
                             rotation_image_paths=[RENDER_DIR/"rotation_0.png", RENDER_DIR/"rotation_1.png"],
                             translation_output_path=VLM_DIR/"translation_constraints.json",
                             rotation_output_path=VLM_DIR/"rotation_constraints.json")
V = get_validation_program(instruction=CONFIG["instruction"], active_object=CONFIG["active"], passive_object=CONFIG["passive"], 
                           active_dimensions=ACTIVE_CATALOG["dimensions"], passive_dimensions=PASSIVE_CATALOG["dimensions"],
                           input_image_paths=[RENDER_DIR/"translation_0.png", RENDER_DIR/"translation_1.png"],
                           output_path=VLM_DIR/"validation_program.json")

# 8. Sample in Mujoco + render successful candidates
T_mujoco_active = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_active_source
T_mujoco_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_passive_source
T_mujoco_goal = T_mujoco_passive @ T_passive_active_edited
scene_builder = SceneBuilder()
scene = scene_builder.build(active_name=CONFIG["active"], passive_name=CONFIG["passive"], 
                    T_mujoco_active=T_mujoco_active, T_mujoco_passive=T_mujoco_passive, T_mujoco_goal=T_mujoco_goal,
                    output_path=XML_DIR/f"{BENCHMARK_NAME}.xml")
digital_twin = DigitalTwin(scene, V, S)
valid_candidates = digital_twin.sample(n_samples=1000, max_valid=5, sim_steps=500, visualize=True) # returns a list of valid T_mujoco_goal

for i, T_mujoco_goal_valid in enumerate(valid_candidates):
    T_passive_active_valid = np.linalg.inv(T_mujoco_passive) @ T_mujoco_goal_valid
    render_for_vlm(active_path=ACTIVE_CATALOG["mesh"], passive_path=PASSIVE_CATALOG["mesh"], output_path=RENDER_DIR/f"valid_candidate_{i}",
                   T_passive_active=T_passive_active_valid, rendered_frame="active",
                   lookat=CONFIG["render_config"][1]["lookat"], front=CONFIG["render_config"][1]["front"], up=CONFIG["render_config"][1]["up"], zoom=CONFIG["render_config"][1]["zoom"])

# 9. Grasp Planning
grasp_planner = GraspPlanner(ACTIVE_CATALOG, camera.get_intrinsics(), ANYGRASP_CHECKPOINT)
candidate_grasps = grasp_planner.plan(source_pose=T_cam_active_source, 
                                      rgb_path=CAMERA_DIR/"source_rgb.png",
                                      depth_path=CAMERA_DIR/"source_depth_m.npy",
                                      mask_path=PERCEPTION_DIR/"source_active_mask.png",
                                      use_anygrasp=False)

# 10. OMPL trajectory (stages: {waypoints + timestamps}) 