from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from pipeline.camera import RGBDFrame, camera_to_world_transform
from pipeline.core import ObjectMask, ObjectPose, REPO_ROOT, make_T, rotz, rpy_to_R


ANYGRASP_DIR = REPO_ROOT.parent / "anygrasp_sdk" / "grasp_detection"
ANYGRASP_CHECKPOINT = ANYGRASP_DIR / "checkpoints" / "checkpoint_detection.tar"
ANYGRASP_ARRAY_WIDTH = 17
ANYGRASP_EEF_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class CameraGrasp:
    """AnyGrasp geometry retained for the camera-aligned choice image."""

    center_camera: np.ndarray
    rotation_camera: np.ndarray
    width_m: float
    depth_m: float
    score: float


@dataclass(frozen=True)
class GraspCandidate:
    """A planner-neutral grasp expressed relative to the active object."""

    key: str
    index: int
    source: str
    T_object_eef: np.ndarray
    hover_offset_object_xyz: np.ndarray
    close_ctrl_value: float
    expand_source_z: bool
    score: float | None = None
    detected_width_m: float | None = None
    camera_grasp: CameraGrasp | None = None


@dataclass(frozen=True)
class GoalFilteredGrasp:
    """A grasp plus the equivalent goal rotations that passed prefiltering."""

    candidate: GraspCandidate
    goal_z_indices: tuple[int, ...]
    goal_q_by_index: tuple[tuple[int, np.ndarray], ...] = ()


@dataclass(frozen=True)
class GraspObservation:
    """RGB-D and pose information needed to run AnyGrasp once per stage."""

    frame: RGBDFrame
    active_mask: ObjectMask
    perceived_source_pose: ObjectPose
    artifact_dir: Path


@dataclass(frozen=True)
class AnyGraspConfig:
    checkpoint_path: Path = field(default_factory=lambda: ANYGRASP_CHECKPOINT)
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    mask_margin_px: int = 10
    max_gripper_width_m: float = 0.05
    gripper_height_m: float = 0.03
    top_down_grasp: bool = False
    top_k: int = 5
    goal_filter_pool_size: int = 10
    hover_distance_m: float = 0.10
    # MuJoCo arm1_box_gripper origin to physical fingertip reach. The AnyGrasp
    # grasp depth is subtracted from this value when aligning the two TCPs.
    eef_fingertip_reach_m: float = 0.1135
    gripper_open_gap_m: float = 0.05
    inference_timeout_s: float = 300.0

    @classmethod
    def from_environment(cls) -> "AnyGraspConfig":
        checkpoint = Path(
            os.getenv("ANYGRASP_CHECKPOINT", str(ANYGRASP_CHECKPOINT))
        ).expanduser()
        return cls(
            checkpoint_path=checkpoint,
            eef_fingertip_reach_m=float(
                os.getenv("ANYGRASP_EEF_FINGERTIP_REACH_M", "0.1135")
            ),
        )


def sample_equivalent_poses_about_z(
    T_world_object: np.ndarray,
    sample_count: int,
) -> list[np.ndarray]:
    """Return evenly spaced poses rotated about the object's local Z axis."""
    T = np.asarray(T_world_object, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T_world_object must be a 4x4 transform")
    count = int(sample_count)
    if count < 1:
        raise ValueError("sample_count must be at least 1")
    poses = []
    for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
        pose = T.copy()
        pose[:3, :3] = T[:3, :3] @ rotz(float(angle))
        poses.append(pose)
    return poses


def authored_grasp_candidates(
    grasp_configs: Sequence[dict],
) -> list[GraspCandidate]:
    """Convert the object catalog's hand-authored entries into candidates."""
    candidates = []
    for index, config in enumerate(grasp_configs):
        hover = config.get(
            "hover_offset_xyz",
            (0.0, 0.0, float(config.get("hover_offset_z", 0.10))),
        )
        candidates.append(
            GraspCandidate(
                key=f"authored-{index}",
                index=index,
                source="authored",
                T_object_eef=make_T(
                    R=rpy_to_R(np.deg2rad(config["offset_rpy_deg"])),
                    p=np.asarray(config["offset_xyz"], dtype=float),
                ),
                hover_offset_object_xyz=np.asarray(hover, dtype=float),
                close_ctrl_value=float(config["close_ctrl_value"]),
                expand_source_z=bool(config.get("expand_source_z", True)),
            )
        )
    return candidates


def point_cloud_from_rgbd(
    frame: RGBDFrame,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the camera-frame AnyGrasp input cloud from aligned RGB-D."""
    if frame.depth_m is None:
        raise RuntimeError("The camera did not return a depth image.")
    if frame.depth_m.shape != frame.rgb_bgr.shape[:2]:
        raise RuntimeError("Color and depth images are not aligned.")
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("Expected 0 < min_depth_m < max_depth_m.")

    depth = np.asarray(frame.depth_m)
    valid = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    if not np.any(valid):
        raise RuntimeError(
            f"No valid depth pixels between {min_depth_m} m and {max_depth_m} m."
        )

    v, u = np.indices(depth.shape, dtype=np.float32)
    z = depth[valid]
    intr = frame.intrinsics
    points = np.column_stack(
        (
            (u[valid] - intr.cx) * z / intr.fx,
            (v[valid] - intr.cy) * z / intr.fy,
            z,
        )
    ).astype(np.float32)
    rgb = cv2.cvtColor(frame.rgb_bgr, cv2.COLOR_BGR2RGB)
    colors = rgb[valid].astype(np.float32) / 255.0
    return points, colors


def _target_mask_for_frame(
    mask: np.ndarray,
    frame: RGBDFrame,
    margin_px: int,
) -> np.ndarray:
    target = np.asarray(mask) > 0
    if target.shape != frame.rgb_bgr.shape[:2]:
        target = cv2.resize(
            target.astype(np.uint8),
            (frame.width, frame.height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    if margin_px > 0:
        size = 2 * int(margin_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        target = cv2.dilate(target.astype(np.uint8), kernel) > 0
    return target


def _grasp_centers_in_mask(
    grasps: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: Sequence[float],
) -> np.ndarray:
    """Return which camera-frame grasp centers project inside a mask."""
    target = np.asarray(target_mask, dtype=bool)
    fx, fy, cx, cy = (float(value) for value in intrinsics)

    translations = grasps[:, 13:16]
    z = translations[:, 2]
    u = np.zeros(len(grasps), dtype=int)
    v = np.zeros(len(grasps), dtype=int)
    in_front = z > 0.0
    u[in_front] = np.rint(
        fx * translations[in_front, 0] / z[in_front] + cx
    ).astype(int)
    v[in_front] = np.rint(
        fy * translations[in_front, 1] / z[in_front] + cy
    ).astype(int)
    in_image = (
        in_front
        & (u >= 0)
        & (u < target.shape[1])
        & (v >= 0)
        & (v < target.shape[0])
    )
    keep = np.zeros(len(grasps), dtype=bool)
    keep[in_image] = target[v[in_image], u[in_image]]
    return keep


def filter_grasp_array_by_mask(
    grasp_array: np.ndarray,
    mask: np.ndarray,
    frame: RGBDFrame,
    margin_px: int,
) -> np.ndarray:
    """Keep grasps whose camera-frame centers project into the active mask."""
    grasps = np.asarray(grasp_array, dtype=float)
    if grasps.ndim != 2 or grasps.shape[1] != ANYGRASP_ARRAY_WIDTH:
        raise ValueError(
            f"Expected an (N, {ANYGRASP_ARRAY_WIDTH}) AnyGrasp array, "
            f"got {grasps.shape}."
        )
    target = _target_mask_for_frame(mask, frame, margin_px)
    intr = frame.intrinsics
    keep = _grasp_centers_in_mask(
        grasps,
        target,
        (intr.fx, intr.fy, intr.cx, intr.cy),
    )
    return grasps[keep]


def _anygrasp_to_candidate(
    row: np.ndarray,
    *,
    rank: int,
    T_world_camera: np.ndarray,
    T_world_object: np.ndarray,
    config: AnyGraspConfig,
) -> GraspCandidate:
    score, width, depth = float(row[0]), float(row[1]), float(row[3])
    R_camera_grasp = np.asarray(row[4:13], dtype=float).reshape(3, 3)
    center_camera = np.asarray(row[13:16], dtype=float)
    T_camera_grasp = make_T(R=R_camera_grasp, p=center_camera)
    T_grasp_eef = make_T(
        R=ANYGRASP_EEF_ROTATION,
        p=np.array([depth - config.eef_fingertip_reach_m, 0.0, 0.0]),
    )
    T_world_eef = T_world_camera @ T_camera_grasp @ T_grasp_eef
    T_object_eef = np.linalg.inv(T_world_object) @ T_world_eef
    hover_offset = -float(config.hover_distance_m) * T_object_eef[:3, 2]
    close_ctrl = float(
        np.clip(config.gripper_open_gap_m - width, 0.0, config.gripper_open_gap_m)
    )
    return GraspCandidate(
        key=f"anygrasp-{rank}",
        index=rank,
        source="anygrasp",
        T_object_eef=T_object_eef,
        hover_offset_object_xyz=hover_offset,
        close_ctrl_value=close_ctrl,
        expand_source_z=False,
        score=score,
        detected_width_m=width,
        camera_grasp=CameraGrasp(
            center_camera=center_camera.copy(),
            rotation_camera=R_camera_grasp.copy(),
            width_m=width,
            depth_m=depth,
            score=score,
        ),
    )


def _run_anygrasp_subprocess(
    points: np.ndarray,
    colors: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: Sequence[float],
    config: AnyGraspConfig,
) -> np.ndarray:
    checkpoint = Path(config.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AnyGrasp checkpoint not found: {checkpoint}")
    with tempfile.TemporaryDirectory(prefix="veriflm_anygrasp_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "input.npz"
        output_path = temp / "grasps.npy"
        runtime_dir = temp / "sdk_runtime"
        runtime_dir.mkdir()
        extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
        gsnet_version = ANYGRASP_DIR / "gsnet_versions" / f"gsnet{extension_suffix}"
        lib_cxx_version = (
            ANYGRASP_DIR.parent
            / "license_registration"
            / "lib_cxx_versions"
            / f"lib_cxx{extension_suffix}"
        )
        gsnet_binary = gsnet_version if gsnet_version.is_file() else ANYGRASP_DIR / "gsnet.so"
        lib_cxx_binary = (
            lib_cxx_version if lib_cxx_version.is_file() else ANYGRASP_DIR / "lib_cxx.so"
        )
        if not gsnet_binary.is_file() or not lib_cxx_binary.is_file():
            raise FileNotFoundError(
                "AnyGrasp ABI-matched gsnet/lib_cxx native modules are unavailable."
            )
        (runtime_dir / f"gsnet{extension_suffix}").symlink_to(gsnet_binary.resolve())
        (runtime_dir / f"lib_cxx{extension_suffix}").symlink_to(lib_cxx_binary.resolve())
        license_dir = ANYGRASP_DIR / "license"
        if license_dir.is_dir():
            (runtime_dir / "license").symlink_to(
                license_dir.resolve(), target_is_directory=True
            )
        np.savez(
            input_path,
            points=points,
            colors=colors,
            target_mask=np.asarray(target_mask, dtype=np.uint8),
            intrinsics=np.asarray(intrinsics, dtype=float),
        )
        command = [
            sys.executable,
            "-m",
            "pipeline.grasp",
            "--anygrasp-worker",
            str(input_path),
            str(output_path),
            str(checkpoint),
            str(float(config.max_gripper_width_m)),
            str(float(config.gripper_height_m)),
            "1" if config.top_down_grasp else "0",
            str(runtime_dir),
        ]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(REPO_ROOT), existing_pythonpath)
            if value
        )
        try:
            result = subprocess.run(
                command,
                cwd=runtime_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(config.inference_timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"AnyGrasp inference timed out after {config.inference_timeout_s:.0f}s."
            ) from exc
        if result.returncode != 0:
            tail = "\n".join((result.stdout or "").splitlines()[-8:])
            detail = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"AnyGrasp worker failed with exit code {result.returncode}.{detail}"
            )
        if not output_path.is_file():
            tail = "\n".join((result.stdout or "").splitlines()[-8:])
            detail = f"\n{tail}" if tail else ""
            raise RuntimeError(
                "AnyGrasp worker exited without producing grasp output."
                f"{detail}"
            )
        grasps = np.asarray(np.load(output_path), dtype=float)
    if grasps.ndim != 2 or grasps.shape[1] != ANYGRASP_ARRAY_WIDTH:
        raise RuntimeError(f"AnyGrasp returned a malformed array: {grasps.shape}")
    if not np.all(np.isfinite(grasps)):
        raise RuntimeError("AnyGrasp returned non-finite grasp values.")
    return grasps


def generate_anygrasp_candidates(
    observation: GraspObservation,
    config: AnyGraspConfig | None = None,
) -> list[GraspCandidate]:
    """Run AnyGrasp, apply the active-object mask, and convert pose frames."""
    config = config or AnyGraspConfig.from_environment()
    points, colors = point_cloud_from_rgbd(
        observation.frame,
        config.min_depth_m,
        config.max_depth_m,
    )
    target_mask = _target_mask_for_frame(
        observation.active_mask.mask,
        observation.frame,
        config.mask_margin_px,
    )
    intr = observation.frame.intrinsics
    intrinsics = (intr.fx, intr.fy, intr.cx, intr.cy)
    raw = _run_anygrasp_subprocess(
        points,
        colors,
        target_mask,
        intrinsics,
        config,
    )
    if len(raw) == 0:
        raise RuntimeError("AnyGrasp produced no collision-free grasps.")
    filtered = filter_grasp_array_by_mask(
        raw,
        target_mask,
        observation.frame,
        0,
    )
    if len(filtered) == 0:
        raise RuntimeError(
            "No AnyGrasp centers projected inside the active-object mask."
        )

    # The worker applies NMS and score sorting. Keep the parent-side ordering
    # explicit as a guard against SDK version differences.
    filtered = filtered[np.argsort(filtered[:, 0])[::-1]]
    T_world_camera = camera_to_world_transform(
        observation.frame.T_camera_to_board,
        observation.frame.T_board_to_world,
    )
    if T_world_camera is None:
        raise RuntimeError("Camera extrinsic is unavailable for AnyGrasp conversion.")
    return [
        _anygrasp_to_candidate(
            row,
            rank=index,
            T_world_camera=T_world_camera,
            T_world_object=observation.perceived_source_pose.T_world_object,
            config=config,
        )
        for index, row in enumerate(filtered)
    ]


def _project_camera_point(
    point: np.ndarray,
    frame: RGBDFrame,
) -> tuple[int, int] | None:
    x, y, z = np.asarray(point, dtype=float)
    if z <= 0.0:
        return None
    intr = frame.intrinsics
    u = int(round(intr.fx * x / z + intr.cx))
    v = int(round(intr.fy * y / z + intr.cy))
    if 0 <= u < frame.width and 0 <= v < frame.height:
        return u, v
    return None


def render_grasp_choices(
    observation: GraspObservation,
    candidates: Sequence[GraspCandidate],
) -> np.ndarray:
    """Render a numbered, camera-aligned image of up to five grasps."""
    image = observation.frame.rgb_bgr.copy()
    mask = np.asarray(observation.active_mask.mask)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    green = np.zeros_like(image)
    green[:, :, 1] = (mask > 0).astype(np.uint8) * 255
    image = cv2.addWeighted(image, 1.0, green, 0.18, 0.0)
    palette = (
        (0, 255, 255),
        (0, 165, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 0),
    )
    for display_index, candidate in enumerate(candidates, start=1):
        grasp = candidate.camera_grasp
        if grasp is None:
            continue
        color = palette[(display_index - 1) % len(palette)]
        center = grasp.center_camera
        jaw = grasp.rotation_camera[:, 1]
        approach = grasp.rotation_camera[:, 0]
        left = center + jaw * grasp.width_m / 2.0
        right = center - jaw * grasp.width_m / 2.0
        behind = center - approach * 0.05
        p_center = _project_camera_point(center, observation.frame)
        p_left = _project_camera_point(left, observation.frame)
        p_right = _project_camera_point(right, observation.frame)
        p_behind = _project_camera_point(behind, observation.frame)
        if p_left is not None and p_right is not None:
            cv2.line(image, p_left, p_right, color, 3, cv2.LINE_AA)
        if p_center is not None and p_behind is not None:
            cv2.arrowedLine(
                image,
                p_behind,
                p_center,
                color,
                3,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        if p_center is not None:
            cv2.circle(image, p_center, 15, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                str(display_index),
                (p_center[0] - 7, p_center[1] + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    legend_height = 42 + 28 * len(candidates)
    legend = np.full((legend_height, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        legend,
        "AnyGrasp goal-feasible choices",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for display_index, candidate in enumerate(candidates, start=1):
        color = palette[(display_index - 1) % len(palette)]
        score = 0.0 if candidate.score is None else candidate.score
        width = 0.0 if candidate.detected_width_m is None else candidate.detected_width_m
        text = f"{display_index}: score={score:.3f}  width={width * 1000.0:.1f} mm"
        cv2.putText(
            legend,
            text,
            (12, 29 + 28 * display_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    return np.vstack((image, legend))


def _read_choice_with_window(window: str, prompt: str) -> str:
    answer: list[str | None] = [None]
    done = threading.Event()

    def read_stdin() -> None:
        try:
            answer[0] = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            answer[0] = "f"
        done.set()

    threading.Thread(target=read_stdin, daemon=True).start()
    while not done.wait(0.05):
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass
    return answer[0] or "f"


def _close_choice_window(window: str) -> None:
    try:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    except cv2.error:
        pass


def choose_anygrasp_candidate(
    observation: GraspObservation,
    candidates: Sequence[GraspCandidate],
    *,
    choice_reader: Callable[[str], str] | None = None,
) -> GraspCandidate | None:
    """Save/show numbered grasps and return the user's 1-based selection."""
    choices = list(candidates)[:5]
    if not choices:
        return None
    preview = render_grasp_choices(observation, choices)
    output_path = Path(observation.artifact_dir) / "anygrasp_top5.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Could not save AnyGrasp preview: {output_path}")
    print(f"[grasp:anygrasp] numbered choices -> {output_path}")

    if choice_reader is None and not sys.stdin.isatty():
        print(
            "[grasp:anygrasp] non-interactive stdin; using hand-authored grasps."
        )
        return None

    window = "AnyGrasp top choices"
    window_open = False
    display_available = (
        not sys.platform.startswith("linux")
        or bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    )
    if choice_reader is None and display_available:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.imshow(window, preview)
            cv2.waitKey(1)
            window_open = True
        except cv2.error as exc:
            print(f"[grasp:anygrasp] could not open preview window: {exc}")

    prompt = f"\nChoose AnyGrasp 1-{len(choices)}, or F for authored fallback: "
    while True:
        if choice_reader is not None:
            try:
                value = choice_reader(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                value = "f"
        elif window_open:
            value = _read_choice_with_window(window, prompt).strip().lower()
        else:
            try:
                value = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                value = "f"
        if value in {"f", "fallback", "n", "none", ""}:
            if window_open:
                _close_choice_window(window)
            print("[grasp:anygrasp] using hand-authored fallback grasps")
            return None
        try:
            selected = int(value) - 1
        except ValueError:
            selected = -1
        if 0 <= selected < len(choices):
            candidate = choices[selected]
            if window_open:
                _close_choice_window(window)
            print(
                f"[grasp:anygrasp] selected {selected + 1} "
                f"(score={candidate.score:.3f})"
            )
            return candidate
        print(f"Enter a number from 1 to {len(choices)}, or F.")


def grasp_candidate_batches(
    *,
    observation: GraspObservation | None,
    anygrasp_config: AnyGraspConfig | None,
    authored_configs: Sequence[dict],
    goal_filter: Callable[
        [Sequence[GraspCandidate], int | None],
        list[GoalFilteredGrasp],
    ],
    choice_reader: Callable[[str], str] | None = None,
) -> Iterator[tuple[str, list[GoalFilteredGrasp]]]:
    """Yield a selected goal-feasible AnyGrasp, then authored fallbacks."""
    if observation is not None:
        try:
            config = anygrasp_config or AnyGraspConfig.from_environment()
            top_k = min(5, max(1, int(config.top_k)))
            # Backfill rejected top-score detections from a small, bounded
            # pool, while stopping the expensive MuJoCo checks at five
            # collision-free choices.
            pool_size = max(top_k, int(config.goal_filter_pool_size))
            generated = generate_anygrasp_candidates(
                observation, config
            )[:pool_size]
            feasible = goal_filter(generated, top_k)
            if feasible:
                selected = choose_anygrasp_candidate(
                    observation,
                    [item.candidate for item in feasible],
                    choice_reader=choice_reader,
                )
                if selected is not None:
                    chosen = next(
                        item for item in feasible if item.candidate.key == selected.key
                    )
                    yield "anygrasp", [chosen]
                    print(
                        "[planning:grasp] selected AnyGrasp did not produce an "
                        "end-to-end plan; trying authored fallback grasps"
                    )
            else:
                print(
                    "[grasp:anygrasp] no generated grasp is collision-free "
                    "at the goal; using authored fallback grasps"
                )
        except Exception as exc:
            print(
                f"[grasp:anygrasp] generation failed ({exc}); "
                "using authored fallback grasps"
            )

    authored = authored_grasp_candidates(authored_configs)
    feasible_authored = goal_filter(authored, None)
    if feasible_authored:
        yield "authored", feasible_authored
    else:
        print(
            "[planning:grasp] no hand-authored grasp is collision-free "
            "at the goal"
        )


def _worker_main(argv: Sequence[str]) -> int:
    if len(argv) != 9 or argv[1] != "--anygrasp-worker":
        return 2
    _, _, input_path, output_path, checkpoint, width, height, top_down, runtime_dir = argv
    # Prefer the ABI-tagged libraries matching the running interpreter over
    # the SDK's generic .so files. Native license failures remain isolated in
    # this subprocess and are reported by the parent as a fallback condition.
    sys.path.insert(0, str(Path(runtime_dir).resolve()))
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]

    from gsnet import AnyGrasp

    data = np.load(input_path)
    points = np.asarray(data["points"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.float32)
    target_mask = np.asarray(data["target_mask"], dtype=bool)
    intrinsics = np.asarray(data["intrinsics"], dtype=float)
    limits = [
        float(value)
        for value in (
            points[:, 0].min(),
            points[:, 0].max(),
            points[:, 1].min(),
            points[:, 1].max(),
            points[:, 2].min(),
            points[:, 2].max(),
        )
    ]
    settings = SimpleNamespace(
        checkpoint_path=str(Path(checkpoint).resolve()),
        max_gripper_width=float(np.clip(float(width), 0.0, 0.1)),
        gripper_height=float(height),
        top_down_grasp=bool(int(top_down)),
        debug=False,
    )
    detector = AnyGrasp(settings)
    detector.load_net()
    grasps, _ = detector.get_grasp(
        points,
        colors,
        lims=limits,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )
    if len(grasps) > 0:
        raw_array = np.asarray(grasps.grasp_group_array, dtype=float)
        keep = _grasp_centers_in_mask(
            raw_array,
            target_mask,
            intrinsics,
        )
        grasps = grasps[keep]
    if len(grasps) > 0:
        grasps = grasps.nms().sort_by_score()
        array = np.asarray(grasps.grasp_group_array, dtype=float)
    else:
        array = np.empty((0, ANYGRASP_ARRAY_WIDTH), dtype=float)
    np.save(output_path, array)
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv))
