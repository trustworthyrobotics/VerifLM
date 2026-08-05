from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from pipeline.core import (
    SamplingConstraints, TaskStage, ValidationProgram, ValidationProgramSelection,
    ValidationProgramSemantics,
    encode_image_as_data_url, extract_json_object, format_prompt_template, prompt_path,
)

DECOMPOSE_TASK_STAGES_PROMPT = prompt_path("decompose_task_stages.txt")
GOAL_IMAGE_EDIT_PROMPT = prompt_path("goal_image_edit.txt")
REASON_ROTATION_PROMPT = prompt_path("reason_sampling_rotation.txt")
REASON_TRANSLATION_PROMPT = prompt_path("reason_sampling_translation.txt")
SELECT_CANDIDATE_PROMPT = prompt_path("select_best_candidate.txt")
VALIDATION_CONFIG_PROMPT = prompt_path("validation_config.txt")

GEMINI_TEXT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation"
)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.5")
OPENAI_VALIDATION_MODEL = os.getenv("OPENAI_VALIDATION_MODEL", "gpt-5.5")
OPENAI_VALIDATION_TIMEOUT_SECONDS = 30.0
OPENAI_VALIDATION_MAX_RETRIES = 2

LOGGER = logging.getLogger(__name__)


def _openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_KEY or OPENAI_API_KEY.")
    return OpenAI(api_key=key)


def _openai_validation_client():
    """Build the bounded OpenAI client used only for validation-program calls."""
    from openai import OpenAI

    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_KEY or OPENAI_API_KEY.")
    return OpenAI(
        api_key=key,
        timeout=OPENAI_VALIDATION_TIMEOUT_SECONDS,
        max_retries=OPENAI_VALIDATION_MAX_RETRIES,
    )


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _object_dimensions(
    dimensions: Sequence[float],
    field_name: str,
) -> tuple[float, float, float]:
    if isinstance(dimensions, (str, bytes)):
        raise ValueError(f"{field_name} must contain exactly three positive numbers.")
    try:
        raw_values = tuple(dimensions)
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must contain exactly three positive numbers."
        ) from exc
    if len(raw_values) != 3:
        raise ValueError(f"{field_name} must contain exactly three positive numbers.")

    values: list[float] = []
    for index, raw_value in enumerate(raw_values):
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError(f"{field_name}[{index}] must be a positive finite number.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name}[{index}] must be a positive finite number."
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name}[{index}] must be a positive finite number.")
        values.append(value)
    return values[0], values[1], values[2]


def _goal_image_bgr(goal_image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if goal_image is None:
        return None
    if not isinstance(goal_image, np.ndarray):
        raise TypeError("goal_image must be a uint8 BGR numpy array or None.")
    if (
        goal_image.dtype != np.uint8
        or goal_image.ndim != 3
        or goal_image.shape[2] != 3
        or goal_image.shape[0] == 0
        or goal_image.shape[1] == 0
    ):
        raise ValueError("goal_image must be a non-empty uint8 BGR image with shape HxWx3.")
    return goal_image


def _validation_request_values(
    instruction: str,
    active_object_name: str,
    active_dimensions: Sequence[float],
    passive_object_name: str,
    passive_dimensions: Sequence[float],
    goal_image: Optional[np.ndarray],
    provider: str,
) -> tuple[
    str,
    str,
    tuple[float, float, float],
    str,
    tuple[float, float, float],
    Optional[np.ndarray],
]:
    if not isinstance(provider, str) or provider.strip().lower() != "openai":
        raise ValueError("Validation-program prediction supports only provider='openai'.")
    return (
        _nonempty_text(instruction, "instruction"),
        _nonempty_text(active_object_name, "active_object_name"),
        _object_dimensions(active_dimensions, "active_dimensions"),
        _nonempty_text(passive_object_name, "passive_object_name"),
        _object_dimensions(passive_dimensions, "passive_dimensions"),
        _goal_image_bgr(goal_image),
    )


def _response_field(value: object, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_refusal(response: object) -> Optional[str]:
    for output in _response_field(response, "output", ()) or ():
        for item in _response_field(output, "content", ()) or ():
            if _response_field(item, "type") != "refusal":
                continue
            reason = str(_response_field(item, "refusal", "")).strip()
            return reason or "The model refused to generate a validation program."
    return None


def _coerce_validation_program(value: object) -> ValidationProgram:
    if isinstance(value, ValidationProgram):
        return value
    validator = getattr(ValidationProgram, "model_validate", None)
    if not callable(validator):
        raise RuntimeError("ValidationProgram must provide Pydantic model_validate().")
    return validator(value)


def _coerce_validation_program_semantics(value: object) -> ValidationProgramSemantics:
    if type(value) is ValidationProgramSemantics:
        return value
    validator = getattr(ValidationProgramSemantics, "model_validate", None)
    if not callable(validator):
        raise RuntimeError(
            "ValidationProgramSemantics must provide Pydantic model_validate()."
        )
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump(mode="python")
    return validator(value)


def _validation_program_semantics_from_json(
    value: object,
) -> ValidationProgramSemantics:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OpenAI returned no validation-program JSON.")
    validator = getattr(ValidationProgramSemantics, "model_validate_json", None)
    if not callable(validator):
        raise RuntimeError(
            "ValidationProgramSemantics must provide Pydantic model_validate_json()."
        )
    # Parse the entire response. Prose-wrapped or fenced JSON is invalid here.
    return validator(value)


def _request_validation_program_openai(
    content: list[dict],
) -> ValidationProgramSemantics:
    client = _openai_validation_client()
    responses = getattr(client, "responses", None)
    if responses is None:
        raise RuntimeError("The installed OpenAI SDK does not provide the Responses API.")

    request_input = [{"role": "user", "content": content}]
    parse = getattr(responses, "parse", None)
    if callable(parse):
        response = parse(
            model=OPENAI_VALIDATION_MODEL,
            input=request_input,
            text_format=ValidationProgramSemantics,
        )
        refusal = _response_refusal(response)
        if refusal is not None:
            raise RuntimeError(f"OpenAI refused the validation-program request: {refusal}")
        parsed = _response_field(response, "output_parsed")
        if parsed is None:
            raise ValueError("OpenAI returned no parsed validation program.")
        return _coerce_validation_program_semantics(parsed)

    create = getattr(responses, "create", None)
    schema_builder = getattr(ValidationProgramSemantics, "model_json_schema", None)
    if not callable(create) or not callable(schema_builder):
        raise RuntimeError(
            "The installed OpenAI SDK cannot request strict validation-program output."
        )
    response = create(
        model=OPENAI_VALIDATION_MODEL,
        input=request_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "validation_program",
                "schema": schema_builder(),
                "strict": True,
            }
        },
    )
    refusal = _response_refusal(response)
    if refusal is not None:
        raise RuntimeError(f"OpenAI refused the validation-program request: {refusal}")
    return _validation_program_semantics_from_json(
        _response_field(response, "output_text")
    )


def _positive_finite_threshold(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a positive finite number.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number.") from exc
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return threshold


def _validation_program_with_thresholds(
    semantics: ValidationProgramSemantics,
    *,
    initial_max_pen: object,
    final_max_pen: object,
) -> ValidationProgram:
    payload = semantics.model_dump(mode="python")
    payload.update(
        initial_max_pen=_positive_finite_threshold(
            initial_max_pen,
            "initial_max_pen",
        ),
        final_max_pen=_positive_finite_threshold(
            final_max_pen,
            "final_max_pen",
        ),
    )
    return _coerce_validation_program(payload)


def predict_validation_program_vlm(
    instruction: str,
    active_object_name: str,
    active_dimensions: Sequence[float],
    passive_object_name: str,
    passive_dimensions: Sequence[float],
    goal_image: Optional[np.ndarray] = None,
    provider: str = "openai",
    *,
    initial_max_pen: float,
    final_max_pen: float,
) -> ValidationProgram:
    """Predict and strictly validate task semantics for candidate validation."""
    initial_max_pen = _positive_finite_threshold(
        initial_max_pen,
        "initial_max_pen",
    )
    final_max_pen = _positive_finite_threshold(
        final_max_pen,
        "final_max_pen",
    )
    (
        instruction,
        active_object_name,
        active_dimensions,
        passive_object_name,
        passive_dimensions,
        goal_image,
    ) = _validation_request_values(
        instruction,
        active_object_name,
        active_dimensions,
        passive_object_name,
        passive_dimensions,
        goal_image,
        provider,
    )
    prompt = format_prompt_template(
        VALIDATION_CONFIG_PROMPT,
        instruction=instruction,
        active_object=active_object_name,
        active_dimensions=list(active_dimensions),
        passive_object=passive_object_name,
        passive_dimensions=list(passive_dimensions),
    )
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    if goal_image is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image_as_data_url(goal_image),
            }
        )
    semantics = _request_validation_program_openai(content)
    return _validation_program_with_thresholds(
        semantics,
        initial_max_pen=initial_max_pen,
        final_max_pen=final_max_pen,
    )


def _error_provenance(exc: Exception) -> str:
    detail = " ".join(str(exc).split()) or "no error detail"
    if len(detail) > 500:
        detail = detail[:497] + "..."
    return f"{type(exc).__name__}: {detail}"


def resolve_validation_program_vlm(
    instruction: str,
    active_object_name: str,
    active_dimensions: Sequence[float],
    passive_object_name: str,
    passive_dimensions: Sequence[float],
    fallback_program: ValidationProgram | dict,
    goal_image: Optional[np.ndarray] = None,
    provider: str = "openai",
) -> ValidationProgramSelection:
    """Return a VLM program, or a prevalidated config fallback with provenance."""
    # A malformed fallback is a configuration error and must never be hidden by
    # the VLM request/fallback boundary.
    fallback = _coerce_validation_program(fallback_program)
    _validation_request_values(
        instruction,
        active_object_name,
        active_dimensions,
        passive_object_name,
        passive_dimensions,
        goal_image,
        provider,
    )

    try:
        program = predict_validation_program_vlm(
            instruction,
            active_object_name,
            active_dimensions,
            passive_object_name,
            passive_dimensions,
            goal_image=goal_image,
            provider=provider,
            initial_max_pen=fallback.initial_max_pen,
            final_max_pen=fallback.final_max_pen,
        )
    except Exception as exc:
        error = _error_provenance(exc)
        LOGGER.warning(
            "Validation-program VLM request failed; "
            "using validated config fallback (%s).",
            error,
        )
        return ValidationProgramSelection(
            program=fallback,
            source="config_fallback",
            error=error,
        )
    return ValidationProgramSelection(program=program, source="vlm", error=None)


def _gemini_client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return genai.Client(api_key=key)


def _gemini_chat(prompt: str, images: Sequence[np.ndarray]) -> str:
    import PIL.Image

    contents: list = [prompt]
    for image in images:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        contents.append(PIL.Image.fromarray(rgb))
    with _gemini_client() as client:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=contents,
        )
    return response.text or ""


def _vision_json(prompt: str, images: Sequence[np.ndarray], provider: str) -> dict:
    images = [image for image in images if image is not None]
    if provider == "gemini":
        return extract_json_object(_gemini_chat(prompt, images))

    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(image)},
            }
        )
    response = _openai_client().chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    return extract_json_object(response.choices[0].message.content or "{}")


def resolve_object_prompts(args) -> tuple[str, str]:
    """Return the catalog prompts for the stage selected before perception."""
    if not args.active_prompt or not args.passive_prompt:
        raise ValueError("A selected stage requires active and passive prompts.")
    return args.active_prompt, args.passive_prompt


def _validate_stages(data: dict, candidates: Sequence[str]) -> list[TaskStage]:
    allowed = set(candidates)
    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("Task decomposition returned no stages.")

    stages: list[TaskStage] = []
    for index, raw in enumerate(raw_stages, start=1):
        active = str(raw.get("active_object") or raw.get("active") or "").strip()
        passive = str(raw.get("passive_object") or raw.get("passive") or "").strip()
        if active not in allowed or passive not in allowed or active == passive:
            raise ValueError(f"Invalid task stage {index}: active={active!r}, passive={passive!r}")
        stages.append(
            TaskStage(
                active_object=active,
                passive_object=passive,
                reason=str(raw.get("reason", "")).strip(),
                instruction=str(raw.get("instruction", "")).strip(),
            )
        )
    return stages


def resolve_task_stages(args, source=None, provider: str = "openai") -> list[TaskStage]:
    candidates = tuple(args.candidate_objects or ())
    if not args.use_vlm_stage_resolution:
        return _validate_stages({"stages": list(args.default_stages)}, candidates)

    available = "\n".join(
        f"- {object_id}: {(args.object_prompts or {}).get(object_id, object_id.replace('_', ' '))}"
        for object_id in candidates
    )
    prompt = format_prompt_template(
        DECOMPOSE_TASK_STAGES_PROMPT,
        instruction=args.instruction,
        available_objects=available,
    )
    images = [] if source is None else [source.rgb_bgr]
    return _validate_stages(_vision_json(prompt, images, provider), candidates)


def prepare_goal_image_vlm(
    source_bgr: np.ndarray,
    instruction: str,
    active_object: str,
    passive_object: Optional[str],
    out_path: str | Path,
    provider: str = "gemini",
) -> np.ndarray:
    if provider != "gemini":
        raise ValueError("The refactored main pipeline uses Gemini for goal-image editing.")

    import PIL.Image
    from google.genai import types

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = format_prompt_template(
        GOAL_IMAGE_EDIT_PROMPT,
        instruction=instruction,
        active_object=active_object,
        passive_object=passive_object or "none",
        revision_feedback="No prior edit-quality feedback.",
    )
    source = PIL.Image.fromarray(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB))
    with _gemini_client() as client:
        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[prompt, source],
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )
    for part in response.candidates[0].content.parts:
        if part.inline_data is None:
            continue
        image = cv2.imdecode(np.frombuffer(part.inline_data.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Gemini returned unreadable image data.")
        cv2.imwrite(str(out_path), image)
        return image
    raise RuntimeError("Gemini returned no edited image.")


def _assessment(value: object) -> str:
    value = str(value).strip().lower()
    return value if value in {"correct", "normal_search", "major_revision"} else "normal_search"


def _direction(value: object) -> str:
    value = str(value).strip().lower()
    return value if value in {"positive", "negative", "none"} else "none"


def reason_sampling_axes_vlm(
    rotation_images: Sequence[np.ndarray],
    translation_images: Sequence[np.ndarray],
    instruction: str,
    active_object_name: str,
    passive_object_name: Optional[str],
    provider: str = "openai",
) -> dict:
    rotation_prompt = format_prompt_template(
        REASON_ROTATION_PROMPT,
        instruction=instruction,
        active_object=active_object_name,
        passive_object=passive_object_name or "none",
    )
    translation_prompt = format_prompt_template(
        REASON_TRANSLATION_PROMPT,
        instruction=instruction,
        active_object=active_object_name,
        passive_object=passive_object_name or "none",
    )
    rotation_raw = _vision_json(rotation_prompt, rotation_images, provider)
    translation_raw = _vision_json(translation_prompt, translation_images, provider)

    rotation = {
        axis: {"assessment": _assessment((rotation_raw.get("rotation") or {}).get(axis, {}).get("assessment"))}
        for axis in ("roll", "pitch", "yaw")
    }
    translation = {
        axis: {
            "assessment": _assessment((translation_raw.get("translation") or {}).get(axis, {}).get("assessment")),
            "direction": _direction((translation_raw.get("translation") or {}).get(axis, {}).get("direction")),
        }
        for axis in ("x", "y", "z")
    }
    return {
        "translation": translation,
        "rotation": rotation,
        "summary": (
            f"Rotation: {rotation_raw.get('summary', '')} "
            f"Translation: {translation_raw.get('summary', '')}"
        ).strip(),
    }


def constraints_from_axis_reasoning(reasoning: dict) -> SamplingConstraints:
    level = {"correct": "low", "normal_search": "medium", "major_revision": "high"}
    translation, rotation = reasoning["translation"], reasoning["rotation"]
    variance = {axis: level[translation[axis]["assessment"]] for axis in ("x", "y", "z")}
    variance.update({axis: level[rotation[axis]["assessment"]] for axis in ("roll", "pitch", "yaw")})
    bias = {
        axis: None
        if translation[axis]["assessment"] == "correct" or translation[axis]["direction"] == "none"
        else translation[axis]["direction"]
        for axis in ("x", "y", "z")
    }
    return SamplingConstraints(
        sampling_variance=variance,
        pos_bias=bias,
        reasoning=reasoning["summary"],
    )


def select_best_candidate_vlm(
    candidate_image_paths: Sequence[str | Path],
    *,
    instruction: str,
    active_object: str,
    passive_object: Optional[str],
    provider: str = "openai",
) -> dict:
    paths = [Path(path) for path in candidate_image_paths]
    if not paths:
        return {"best_index": -1, "reasoning": "No valid candidate images."}

    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    prompt = format_prompt_template(
        SELECT_CANDIDATE_PROMPT,
        instruction=instruction,
        active_object=active_object,
        passive_object=passive_object or "none",
    )
    prompt += "\n" + "\n".join(f"candidate_{i}" for i in range(len(images)))
    data = _vision_json(prompt, images, provider)
    index = max(0, min(int(data.get("best_index", 0)), len(paths) - 1))
    return {"best_index": index, "reasoning": str(data.get("reasoning", ""))}
