from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from pydantic import TypeAdapter

from pipeline.core import (
    PositiveFiniteFloat, SamplingConstraints, TaskStage, ValidationProgram,
    ValidationProgramSelection,
    ValidationProgramSemantics,
    encode_image_as_data_url, extract_json_object, format_prompt_template, prompt_path,
)

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
DIMENSIONS_ADAPTER = TypeAdapter(
    tuple[PositiveFiniteFloat, PositiveFiniteFloat, PositiveFiniteFloat]
)
THRESHOLD_ADAPTER = TypeAdapter(PositiveFiniteFloat)


def _openai_api_key() -> str:
    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_KEY or OPENAI_API_KEY.")
    return key


def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=_openai_api_key())


def _openai_validation_client():
    """Build the bounded OpenAI client used only for validation-program calls."""
    from openai import OpenAI

    return OpenAI(
        api_key=_openai_api_key(),
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
    try:
        return DIMENSIONS_ADAPTER.validate_python(dimensions)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must contain exactly three positive numbers."
        ) from exc


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
    return ValidationProgram.model_validate(value)


def _coerce_validation_program_semantics(value: object) -> ValidationProgramSemantics:
    if type(value) is ValidationProgramSemantics:
        return value
    if isinstance(value, ValidationProgramSemantics):
        value = value.model_dump(mode="python")
    return ValidationProgramSemantics.model_validate(value)


def _request_validation_program_openai(
    content: list[dict],
) -> ValidationProgramSemantics:
    client = _openai_validation_client()
    response = client.responses.parse(
        model=OPENAI_VALIDATION_MODEL,
        input=[{"role": "user", "content": content}],
        text_format=ValidationProgramSemantics,
    )
    refusal = _response_refusal(response)
    if refusal is not None:
        raise RuntimeError(f"OpenAI refused the validation-program request: {refusal}")
    parsed = _response_field(response, "output_parsed")
    if parsed is None:
        raise ValueError("OpenAI returned no parsed validation program.")
    return _coerce_validation_program_semantics(parsed)


def _positive_finite_threshold(value: object, field_name: str) -> float:
    try:
        return THRESHOLD_ADAPTER.validate_python(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive finite number.") from exc


def _validation_program_with_thresholds(
    semantics: ValidationProgramSemantics,
    *,
    initial_max_pen: object,
    final_max_pen: object,
) -> ValidationProgram:
    return ValidationProgram.model_validate(
        {
            **semantics.model_dump(mode="python"),
            "initial_max_pen": initial_max_pen,
            "final_max_pen": final_max_pen,
        }
    )


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


def resolve_task_stages(args) -> list[TaskStage]:
    return [
        TaskStage(
            active_object=stage["active"],
            passive_object=stage["passive"],
            reason=stage.get("reason", ""),
            instruction=stage.get("instruction", ""),
        )
        for stage in args.default_stages
    ]


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
