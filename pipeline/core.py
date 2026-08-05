from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.camera import RGBDFrame


# Repository layout and stage artifacts
REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"
EXTRINSICS_DIR = REPO_ROOT / "extrinsics"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class StagePaths:
    inputs: Path
    outputs: Path
    depth: Path
    masks: Path
    foundation_pose: Path
    candidates: Path
    debug: Path
    trajectories: Path

    @classmethod
    def create(
        cls,
        benchmark: str,
        *,
        stage_index: int = 0,
        stage_count: int = 1,
    ) -> "StagePaths":
        root = ARTIFACTS_DIR / benchmark
        stage = f"stage_{stage_index + 1:02d}" if stage_count > 1 else None

        def child(name: str) -> Path:
            path = root / name
            return path / stage if stage else path

        paths = cls(
            inputs=child("inputs"),
            outputs=child("outputs"),
            depth=child("depth"),
            masks=child("masks"),
            foundation_pose=child("foundation_pose"),
            candidates=child("candidates"),
            debug=child("debug"),
            trajectories=child("trajectories"),
        )
        for path in paths.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return paths


def model_path(relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    # Config paths currently include a leading "models/".
    parts = path.parts[1:] if path.parts and path.parts[0] == "models" else path.parts
    return MODELS_DIR.joinpath(*parts)


def prompt_path(filename: str) -> Path:
    return PROMPTS_DIR / filename


def extrinsic_path(camera_type: str) -> Path:
    normalized = camera_type.lower().replace("-", "_")
    filename = "extrinsics_orbbec.txt" if normalized in {"gemini", "orbbec"} else "extrinsics_realsense.txt"
    return EXTRINSICS_DIR / filename

# Shared pipeline data
@dataclass
class ObjectMask:
    name: str
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    score: float


@dataclass
class ObjectPose:
    name: str
    T_world_object: np.ndarray
    confidence: float = 0.0
    note: str = ""




@dataclass
class TaskStage:
    active_object: str
    passive_object: str
    reason: str = ""
    instruction: str = ""


@dataclass
class SamplingConstraints:
    sampling_variance: dict[str, str]
    pos_bias: dict[str, Optional[str]]
    reasoning: str


NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
OrientationDriftDegrees = Annotated[
    float,
    Field(strict=True, ge=0.0, le=180.0, allow_inf_nan=False),
]


class _StrictJsonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json_dict(self) -> dict:
        return self.model_dump(mode="json")


class XYZRelation(_StrictJsonModel):
    axis: Literal["x", "y", "z"]
    relation: Literal["greater_than", "less_than"]
    subject: Literal["active"]
    reference: Literal["passive"]


class ValidationProgramSemantics(_StrictJsonModel):
    validation_mode: Literal["settling", "geometric_only"]
    required_contacts: list[Literal["active_passive", "active_table"]]
    forbidden_contacts: list[Literal["active_passive", "active_table"]]
    xyz_relations: list[XYZRelation]
    allowed_position_drift_fraction: Annotated[
        list[NonNegativeFiniteFloat],
        Field(min_length=3, max_length=3),
    ] | None
    allowed_orientation_drift_deg: OrientationDriftDegrees | None
    reasoning: str

    @field_validator("required_contacts", "forbidden_contacts")
    @classmethod
    def reject_duplicate_contacts(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("contact lists cannot contain duplicates")
        return value

    @field_validator("reasoning")
    @classmethod
    def require_reasoning(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reasoning must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_program_semantics(self):
        overlap = set(self.required_contacts) & set(self.forbidden_contacts)
        if overlap:
            raise ValueError(f"contacts cannot be both required and forbidden: {sorted(overlap)}")
        has_position_drift = self.allowed_position_drift_fraction is not None
        has_orientation_drift = self.allowed_orientation_drift_deg is not None
        if self.validation_mode == "settling" and not (has_position_drift and has_orientation_drift):
            raise ValueError("settling mode requires both drift tolerances")
        if self.validation_mode == "geometric_only" and (has_position_drift or has_orientation_drift):
            raise ValueError("geometric_only mode must not define drift tolerances")
        return self


class ValidationProgram(ValidationProgramSemantics):
    initial_max_pen: PositiveFiniteFloat
    final_max_pen: PositiveFiniteFloat


class ValidationProgramSelection(_StrictJsonModel):
    program: ValidationProgram
    source: Literal["vlm", "config_fallback"]
    error: str | None = None


@dataclass
class PerceptionResult:
    source: RGBDFrame
    goal_bgr: np.ndarray
    goal_depth_m: np.ndarray
    masks: dict[str, ObjectMask]
    source_active: ObjectPose
    source_passive: ObjectPose
    goal_active: ObjectPose
    goal_passive: ObjectPose
    relative_goal: np.ndarray


@dataclass
class DigitalTwinResult:
    xml_path: Path
    model: object
    data: object
    source_pose_mj: ObjectPose
    passive_pose_mj: ObjectPose
    initial_goal_pose_mj: ObjectPose


@dataclass
class PipelineArtifacts:
    source: RGBDFrame
    goal: np.ndarray
    source_pose: ObjectPose
    raw_goal_pose: ObjectPose
    refined_goal_pose: ObjectPose
    trajectory: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class GoalRefinementResult:
    twin: object
    refined_goal: ObjectPose
    constraints: SamplingConstraints
    selection: dict
    validation_program_selection: ValidationProgramSelection

# Rigid-transform helpers
def rotx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def roty(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_R(rpy) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def make_T(R=None, p=None) -> np.ndarray:
    T = np.eye(4, dtype=float)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float)
    return T


def mat_to_quat_wxyz(R) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    R = np.asarray(R, dtype=float)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x, y, z = 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x, y, z = (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x, y, z = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / np.linalg.norm(quat)


def quat_wxyz_to_mat(quat) -> np.ndarray:
    """Convert a normalized wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rpy_to_quat(roll, pitch, yaw) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def T_from_pos_quat(pos, quat) -> np.ndarray:
    return make_T(R=quat_wxyz_to_mat(quat), p=pos)


def matrix_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return tuple(float(np.degrees(v)) for v in (roll, pitch, yaw))

# Prompt and JSON helpers
def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def format_prompt_template(path: str | Path, **kwargs) -> str:
    template = load_prompt_template(path)
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        def replace(match):
            key = match.group(1)
            if key not in kwargs:
                raise KeyError(key)
            return str(kwargs[key])
        return re.sub(r"\{([A-Za-z0-9_]+)\}", replace, template)


def encode_image_as_data_url(image_bgr: np.ndarray, suffix: str = ".png") -> str:
    ok, buf = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    mime = "image/png" if suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])

# Artifact persistence
def save_stage_poses(benchmark: str, poses: dict) -> None:
    path = ARTIFACTS_DIR / benchmark / "poses" / "stage_poses.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poses, indent=2), encoding="utf-8")


def save_source(paths: StagePaths, source) -> None:
    cv2.imwrite(str(paths.inputs / "source_rgb.png"), source.rgb_bgr)
    if source.depth_m is not None:
        np.save(paths.inputs / "source_depth_m.npy", source.depth_m.astype(np.float32))


def save_pipeline_result(paths: StagePaths, artifacts: PipelineArtifacts) -> None:
    data = {
        "source_pose": artifacts.source_pose.T_world_object.tolist(),
        "raw_goal_pose": artifacts.raw_goal_pose.T_world_object.tolist(),
        "refined_goal_pose": artifacts.refined_goal_pose.T_world_object.tolist(),
        "trajectory": artifacts.trajectory,
        "metadata": artifacts.metadata,
    }
    (paths.outputs / "pipeline_result.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
