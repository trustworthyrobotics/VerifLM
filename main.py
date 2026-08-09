from config import OBJECT_CATALOG, BENCHMARK_CONFIGS
from pathlib import Path
import argparse
from camera import Camera
from vlm import edit_image
from perception import DepthEstimator, BBoxer, Segmenter

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

GROUNDING_DINO_DIR = ROOT_DIR.parent / "GroundingDINO"
SAM2_DIR = ROOT_DIR.parent / "sam2"
FP_DIR = ROOT_DIR.parent / "FoundationPose"

# 1. Camera capture RGBD
# Output: rgbd observations stored in CAMERA_DIR
camera = Camera()
camera.start()
camera.capture(render=True, save=True, calculate_extrinsics=args.calc_extrinsics, out_dir=CAMERA_DIR)
camera.stop()

# 2. Query VLM to edit the image
# Output: edited image stored in VLM_DIR
edit_image(instruction=CONFIG["instruction"], active=CONFIG["active"], passive=CONFIG["passive"], input_image_path=CAMERA_DIR/"source_rgb.png", output_image_path=VLM_DIR/"edited_rgb.png")

# 3. VGGT depth estimation
# Output: estimated depths stored in PERCEPTION_DIR
depth_estimator = DepthEstimator()
depth_estimator.predict(input_paths=[CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png"], output_paths=[PERCEPTION_DIR/"source_vggt_depth_raw.npy", PERCEPTION_DIR/"edited_vggt_depth_raw.npy"])

# 4. GroundingDINO + SA2 loop (user provides bbox if necessary) + depth correction
# Output: bboxes and masks stored in PERCEPTION_DIR
bboxer = BBoxer(GROUNDING_DINO_DIR)
bboxes_json = bboxer.predict(input_paths=[CAMERA_DIR/"source_rgb.png", CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png", VLM_DIR/"edited_rgb.png"], 
                            prompts=[ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"], ACTIVE_CATALOG["prompt"], PASSIVE_CATALOG["prompt"]],
                            output_path=PERCEPTION_DIR/"bboxes.json")
segmenter = Segmenter(SAM2_DIR)
segmenter.predict(input_paths=[CAMERA_DIR/"source_rgb.png", CAMERA_DIR/"source_rgb.png", VLM_DIR/"edited_rgb.png", VLM_DIR/"edited_rgb.png"],
                  bboxes=list(bboxes_json.values()), 
                  output_paths=[PERCEPTION_DIR/"source_active_mask.png", PERCEPTION_DIR/"source_passive_mask.png", PERCEPTION_DIR/"edited_active_mask.png", PERCEPTION_DIR/"edited_passive_mask.png"])
depth_estimator.correct(depth_paths=[PERCEPTION_DIR/"source_vggt_depth_raw.npy", PERCEPTION_DIR/"edited_vggt_depth_raw.npy"],
                        mask_paths=[PERCEPTION_DIR/"source_passive_mask.png", PERCEPTION_DIR/"edited_passive_mask.png"], 
                        output_path=PERCEPTION_DIR/"edited_vggt_depth_corrected.npy")

# 5. FP pose estimation + render image in open3d

# 1. Make camera able to load extrinsics!!!

# 6. Query VLM for validation program V and sampling constraints S
# 7. Sample in Mujoco + render successful candidates
# 8. Grasp Planning
# 9. OMPL trajectory (stages: {waypoints + timestamps}) 