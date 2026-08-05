from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np

from pipeline.camera import RGBDFrame, camera_to_world_transform, make_depth_vis
from pipeline.core import REPO_ROOT, StagePaths, model_path, ObjectMask, ObjectPose, PerceptionResult


FOUNDATIONPOSE_REPO = Path(
    os.getenv("FOUNDATIONPOSE_REPO", str(REPO_ROOT.parent / "FoundationPose"))
).expanduser().resolve()
GROUNDING_DINO_REPO = Path(
    os.getenv("GROUNDING_DINO_REPO", str(REPO_ROOT.parent / "GroundingDINO"))
).expanduser().resolve()
SAM2_REPO = Path(os.getenv("SAM2_REPO", str(REPO_ROOT.parent / "sam2"))).expanduser().resolve()

def _add_repo(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def predict_goal_depth(goal_bgr: np.ndarray, out_dir: Path) -> np.ndarray:
    """Run VGGT and return its dense goal-image depth estimate."""
    repo = os.getenv("VGGT_REPO")
    if repo:
        _add_repo(Path(repo).expanduser().resolve())

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = goal_bgr.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "goal.png"
        cv2.imwrite(str(path), goal_bgr)
        images = load_and_preprocess_images([str(path)]).to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=dtype):
                    prediction = model(images)
            else:
                prediction = model(images)

    for key in ("depth", "depth_map", "depth_maps"):
        if key in prediction:
            depth = prediction[key]
            break
    else:
        raise KeyError(f"VGGT returned no depth map: {list(prediction)}")

    if torch.is_tensor(depth):
        depth = depth.detach().float().cpu().numpy()
    del model, prediction, images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    depth_m = np.squeeze(depth).astype(np.float32)
    while depth_m.ndim > 2:
        depth_m = depth_m[0]
    if depth_m.shape != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = np.nan

    np.save(out_dir / "goal_depth_m.npy", depth_m)
    cv2.imwrite(
        str(out_dir / "goal_depth_mm.png"),
        np.clip(np.nan_to_num(depth_m) * 1000.0, 0, 65535).astype(np.uint16),
    )
    cv2.imwrite(str(out_dir / "goal_depth_vis.png"), make_depth_vis(depth_m))
    return depth_m


def _load_gdino():
    _add_repo(GROUNDING_DINO_REPO)
    from groundingdino.util.inference import load_model

    return load_model(
        str(GROUNDING_DINO_REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        str(GROUNDING_DINO_REPO / "weights/groundingdino_swint_ogc.pth"),
    )


def _load_sam2():
    _add_repo(SAM2_REPO)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(SAM2_REPO / "checkpoints/sam2.1_hiera_large.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return SAM2ImagePredictor(model)


def _segment(
    gdino,
    sam2,
    image_bgr: np.ndarray,
    prompt: str,
    name: str,
    out_dir: Path,
) -> ObjectMask:
    from groundingdino.util.inference import load_image, predict

    input_path = out_dir / f"{name}_input.png"
    cv2.imwrite(str(input_path), image_bgr)
    _, image = load_image(str(input_path))
    boxes, scores, _ = predict(
        model=gdino,
        image=image,
        caption=prompt,
        box_threshold=0.35,
        text_threshold=0.25,
    )
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO found no object for {prompt!r}.")

    index = int(scores.argmax())
    height, width = image_bgr.shape[:2]
    cx, cy, bw, bh = boxes[index].tolist()
    bbox = (
        int((cx - bw / 2) * width),
        int((cy - bh / 2) * height),
        int((cx + bw / 2) * width),
        int((cy + bh / 2) * height),
    )
    sam2.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    masks, mask_scores, _ = sam2.predict(
        box=np.asarray(bbox, dtype=float),
        multimask_output=True,
    )
    mask = (masks[int(np.argmax(mask_scores))] > 0).astype(np.uint8) * 255
    cv2.imwrite(str(out_dir / f"{name}_mask.png"), mask)
    return ObjectMask(name=name, mask=mask, bbox_xyxy=bbox, score=float(scores[index]))


def segment_scene(
    source_bgr: np.ndarray,
    goal_bgr: np.ndarray,
    active_prompt: str,
    passive_prompt: str,
    out_dir: Path,
) -> dict[str, ObjectMask]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*_mask.png"):
        stale.unlink()
    gdino, sam2 = _load_gdino(), _load_sam2()
    requests = (
        ("source_active", source_bgr, active_prompt),
        ("source_passive", source_bgr, passive_prompt),
        ("goal_active", goal_bgr, active_prompt),
        ("goal_passive", goal_bgr, passive_prompt),
    )
    masks = {name: _segment(gdino, sam2, image, prompt, name, out_dir) for name, image, prompt in requests}
    del gdino, sam2
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return masks


def correct_goal_depth(
    source_depth: np.ndarray,
    goal_depth: np.ndarray,
    source_passive_mask: np.ndarray,
    goal_passive_mask: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Correct VGGT scale and Z offset using passive-object dense correspondences."""
    h, w = source_depth.shape[:2]
    source_mask = cv2.resize(source_passive_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    goal_mask = cv2.resize(goal_passive_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(source_mask & goal_mask)

    if len(xs) < 20:
        source_values = source_depth[source_mask]
        goal_values = goal_depth[goal_mask]
        source_values = source_values[np.isfinite(source_values) & (source_values > 0)]
        goal_values = goal_values[np.isfinite(goal_values) & (goal_values > 0)]
        scale = float(np.median(source_values)) / max(float(np.median(goal_values)), 1e-6)
        corrected = goal_depth * scale
        return corrected.astype(np.float32), scale, 0.0

    source_z = source_depth[ys, xs].astype(np.float64)
    goal_z = goal_depth[ys, xs].astype(np.float64)
    valid = np.isfinite(source_z) & np.isfinite(goal_z) & (source_z > 0.05) & (goal_z > 0.05)
    xs, ys, source_z, goal_z = xs[valid], ys[valid], source_z[valid], goal_z[valid]
    if len(xs) > 8000:
        indices = np.random.default_rng(0).choice(len(xs), 8000, replace=False)
        xs, ys, source_z, goal_z = xs[indices], ys[indices], source_z[indices], goal_z[indices]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    unproject = lambda z: np.stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z), axis=1)
    source_points, goal_points = unproject(source_z), unproject(goal_z)
    source_mean, goal_mean = source_points.mean(0), goal_points.mean(0)
    source_centered, goal_centered = source_points - source_mean, goal_points - goal_mean
    variance = float(np.mean(np.sum(goal_centered**2, axis=1)))
    if variance < 1e-10:
        scale = float(np.median(source_z)) / max(float(np.median(goal_z)), 1e-6)
        return (goal_depth * scale).astype(np.float32), scale, 0.0

    covariance = source_centered.T @ goal_centered / len(goal_points)
    U, singular_values, Vt = np.linalg.svd(covariance)
    sign = float(np.sign(np.linalg.det(U @ Vt)))
    correction = np.diag([1.0, 1.0, sign])
    rotation = U @ correction @ Vt
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = source_mean - scale * (rotation @ goal_mean)
    shift = float(translation[2])
    if not 0.3 <= scale <= 3.0:
        scale = float(np.median(source_z)) / max(float(np.median(goal_z)), 1e-6)
        shift = 0.0
    corrected = np.clip(goal_depth * scale + shift, 0.0, None).astype(np.float32)
    corrected[~np.isfinite(goal_depth)] = np.nan
    return corrected, scale, shift


def _wait_vram_free(container: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["docker", "exec", container, "pgrep", "-f", "python"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return
        time.sleep(1.0)


def run_foundationpose(
    *,
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    mesh_path: Path,
    object_name: str,
    mask: np.ndarray,
    T_camera_to_world: np.ndarray,
    K: np.ndarray,
    out_dir: Path,
) -> ObjectPose:
    container = os.getenv("FP_CONTAINER", "foundationpose_realsense")
    work = (out_dir / object_name.replace(" ", "_")).resolve()
    result_dir = work / "fp_out"
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "exec", container, "rm", "-rf", str(result_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "exec", container, "pkill", "-9", "-f", "python"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_vram_free(container)

    color_path, depth_path, mask_path = work / "color.png", work / "depth.npy", work / "mask.png"
    cv2.imwrite(str(color_path), rgb_bgr)
    np.save(depth_path, np.ascontiguousarray(depth_m, dtype=np.float32))
    cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)

    command = [
        "docker", "exec", container,
        "python", str(FOUNDATIONPOSE_REPO / "run_frame.py"),
        "--mesh_file", str(mesh_path.resolve()),
        "--color_file", str(color_path),
        "--depth_file", str(depth_path),
        "--mask_file", str(mask_path),
        "--fx", str(float(K[0, 0])),
        "--fy", str(float(K[1, 1])),
        "--cx", str(float(K[0, 2])),
        "--cy", str(float(K[1, 2])),
        "--debug", "1",
        "--debug_dir", str(result_dir),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    subprocess.run(
        ["docker", "exec", container, "pkill", "-9", "-f", "python"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_vram_free(container)
    if result.returncode != 0:
        tail = "\n".join(result.stdout.decode(errors="replace").splitlines()[-8:])
        raise RuntimeError(f"FoundationPose failed for {object_name}:\n{tail}")

    pose_path = result_dir / "ob_in_cam" / "000000.txt"
    if not pose_path.exists():
        raise RuntimeError(f"FoundationPose produced no pose for {object_name}.")
    T_object_camera = np.loadtxt(pose_path)
    return ObjectPose(
        name=object_name,
        T_world_object=T_camera_to_world @ T_object_camera,
        confidence=0.8,
        note="FoundationPose",
    )


def _serialize_pose(pose: ObjectPose) -> dict:
    return {
        "name": pose.name,
        "T": pose.T_world_object.tolist(),
        "confidence": pose.confidence,
        "note": pose.note,
    }


class PerceptionPipeline:
    """Owns goal depth, segmentation, FoundationPose, and passive-relative correction."""

    def run(
        self,
        *,
        source: RGBDFrame,
        goal_bgr: np.ndarray,
        active_prompt: str,
        passive_prompt: str,
        active_mesh: str,
        passive_mesh: str,
        paths: StagePaths,
        debug=None,
        progress=None,
        camera_views=None,
    ) -> PerceptionResult:
        if source.depth_m is None:
            raise RuntimeError("Source RGB-D frame has no depth.")

        if progress is not None:
            progress.step(3, "computing goal depth with VGGT")
        goal_depth = predict_goal_depth(goal_bgr, paths.depth)

        height, width = source.rgb_bgr.shape[:2]
        goal_for_pose = cv2.resize(goal_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        if goal_depth.shape != (height, width):
            goal_depth = cv2.resize(goal_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        if progress is not None:
            progress.step(4, "running segmentation masks")
        masks = segment_scene(
            source.rgb_bgr,
            goal_for_pose,
            active_prompt,
            passive_prompt,
            paths.masks,
        )
        if debug is not None:
            masks = debug.review_masks(
                source_bgr=source.rgb_bgr,
                source_depth=source.depth_m,
                goal_bgr=goal_for_pose,
                goal_depth=goal_depth,
                camera_matrix=source.camera_matrix,
                masks=masks,
                mask_dir=paths.masks,
                load_sam2_predictor=_load_sam2,
            )
        if progress is not None:
            progress.step(5, "correcting goal depth and running FoundationPose")
        corrected_depth, scale, shift = correct_goal_depth(
            source.depth_m,
            goal_depth,
            masks["source_passive"].mask,
            masks["goal_passive"].mask,
            source.camera_matrix,
        )
        np.save(paths.depth / "goal_depth_corrected_m.npy", corrected_depth)
        (paths.depth / "depth_correction.json").write_text(
            json.dumps({"scale": scale, "shift_m": shift}, indent=2), encoding="utf-8"
        )

        T_camera_to_world = camera_to_world_transform(
            source.T_camera_to_board, source.T_board_to_world
        )
        if T_camera_to_world is None:
            raise RuntimeError("Camera extrinsic is unavailable.")
        active_mesh_path, passive_mesh_path = model_path(active_mesh), model_path(passive_mesh)
        requests = {
            "source_active": (source.rgb_bgr, source.depth_m, active_mesh_path, active_prompt),
            "source_passive": (source.rgb_bgr, source.depth_m, passive_mesh_path, passive_prompt),
            "goal_active": (goal_for_pose, corrected_depth, active_mesh_path, active_prompt),
            "goal_passive": (goal_for_pose, corrected_depth, passive_mesh_path, passive_prompt),
        }
        poses = {
            key: run_foundationpose(
                rgb_bgr=image,
                depth_m=depth,
                mesh_path=mesh,
                object_name=f"{key}_{name}",
                mask=masks[key].mask,
                T_camera_to_world=T_camera_to_world,
                K=source.camera_matrix,
                out_dir=paths.foundation_pose,
            )
            for key, (image, depth, mesh, name) in requests.items()
        }
        (paths.foundation_pose / "poses.json").write_text(
            json.dumps({key: _serialize_pose(value) for key, value in poses.items()}, indent=2),
            encoding="utf-8",
        )
        corrected_goal_passive = poses["goal_passive"].T_world_object.copy()
        corrected_goal_passive[:3, :3] = poses["source_passive"].T_world_object[:3, :3]
        relative_goal = np.linalg.inv(corrected_goal_passive) @ poses["goal_active"].T_world_object

        if debug is not None:
            debug.preview_foundationpose_scenes(
                active_mesh=active_mesh_path,
                passive_mesh=passive_mesh_path,
                source_active_pose=poses["source_active"],
                source_passive_pose=poses["source_passive"],
                goal_active_pose=poses["goal_active"],
                goal_passive_pose=corrected_goal_passive,
                camera_views=camera_views,
            )
        return PerceptionResult(
            source=source,
            goal_bgr=goal_for_pose,
            goal_depth_m=corrected_depth,
            masks=masks,
            source_active=poses["source_active"],
            source_passive=poses["source_passive"],
            goal_active=poses["goal_active"],
            goal_passive=poses["goal_passive"],
            relative_goal=relative_goal,
        )
