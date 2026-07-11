from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from pipeline.core import (
    SamplingConstraints, TaskStage,
    encode_image_as_data_url, extract_json_object, format_prompt_template, prompt_path,
)

DECOMPOSE_TASK_STAGES_PROMPT = prompt_path("decompose_task_stages.txt")
GOAL_IMAGE_EDIT_PROMPT = prompt_path("goal_image_edit.txt")
REASON_ROTATION_PROMPT = prompt_path("reason_sampling_rotation.txt")
REASON_TRANSLATION_PROMPT = prompt_path("reason_sampling_translation.txt")
SELECT_CANDIDATE_PROMPT = prompt_path("select_best_candidate.txt")

GEMINI_TEXT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation"
)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.5")


def _openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_KEY or OPENAI_API_KEY.")
    return OpenAI(api_key=key)


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
    contact_target = str(translation_raw.get("contact_target", "passive")).lower()
    if contact_target not in {"passive", "table"}:
        contact_target = "passive"
    return {
        "translation": translation,
        "rotation": rotation,
        "contact_target": contact_target,
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
        contact_target=reasoning["contact_target"],
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
