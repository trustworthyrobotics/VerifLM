"""Post-hoc true-mesh candidate renders for an estimated-mesh run."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BENCHMARK_CONFIGS, OBJECT_CATALOG, T_mujoco_board
from src.perception import PoseEstimator
from src.render import render_for_vlm


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--reuse-true-poses", action="store_true")
    args = parser.parse_args()

    config = BENCHMARK_CONFIGS[args.benchmark]
    active, passive = config["active"][args.stage], config["passive"][args.stage]
    base = ROOT / "artifacts" / f"{args.benchmark}_stage{args.stage}"
    camera, estimated = base / "camera", base / "estimated"
    perception, dtwin, render = estimated / "perception", estimated / "dtwin", estimated / "render"
    debug = estimated / "true_mesh_debug"
    debug.mkdir(parents=True, exist_ok=True)

    K = np.asarray(json.loads((camera / "source_camera_meta.json").read_text())["K"])
    estimated_active = np.load(perception / "source_active_pose_cam.npy")
    estimated_passive = np.load(perception / "source_passive_pose_cam.npy")
    true_active_path, true_passive_path = debug / "source_active_true_pose_cam.npy", debug / "source_passive_true_pose_cam.npy"

    if args.reuse_true_poses:
        true_active, true_passive = np.load(true_active_path), np.load(true_passive_path)
    else:
        estimator = PoseEstimator(ROOT.parent / "FoundationPose")
        true_active = estimator.predict(camera / "source_rgb.png", camera / "source_depth_m.npy",
            perception / "source_active_mask.png", path(OBJECT_CATALOG[active]["mesh"]), K, debug, true_active_path.name)
        true_passive = estimator.predict(camera / "source_rgb.png", camera / "source_depth_m.npy",
            perception / "source_passive_mask.png", path(OBJECT_CATALOG[passive]["mesh"]), K, debug, true_passive_path.name)

    M_active = np.linalg.inv(estimated_active) @ true_active
    M_passive = np.linalg.inv(estimated_passive) @ true_passive
    np.savez(debug / "mesh_frame_alignment.npz", active=M_active, passive=M_passive)

    T_cam_board = np.load(ROOT / "artifacts" / "calibration" / "extrinsics.npy")
    T_world_passive_est = T_mujoco_board @ np.linalg.inv(T_cam_board) @ estimated_passive
    view = config["render_config"][1]
    candidates = sorted(dtwin.glob("valid_candidate_*.npy"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))[:5]
    if not candidates:
        raise FileNotFoundError(f"No candidates in {dtwin}")
    for candidate in candidates:
        T_est = np.linalg.inv(T_world_passive_est) @ np.load(candidate)
        T_true = np.linalg.inv(M_passive) @ T_est @ M_active
        output = render / f"{candidate.stem}_true.png"
        render_for_vlm(path(OBJECT_CATALOG[active]["mesh"]), path(OBJECT_CATALOG[passive]["mesh"]), output,
            T_true, "active", lookat=view["lookat"], front=view["front"], up=view["up"], zoom=view["zoom"])
        print(f"[true-mesh]: {output}")


if __name__ == "__main__":
    main()
