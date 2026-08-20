import base64
import json
from pathlib import Path
from typing import Literal

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from src.digital_twin import SamplingConstraints, ValidationProgram, XYZRelation
from src.digital_twin import POSITION_TOLERANCE_MAP, POSITION_CONSTRAINT_MAP, ORIENTATION_TOLERANCE_MAP, ORIENTATION_CONSTRAINT_MAP

load_dotenv()
openai_client = OpenAI()
gemini_client = genai.Client()

def png_to_b64(png_path):
    return base64.b64encode(Path(png_path).read_bytes()).decode('utf-8')

def load_prompt(path, **kwargs):
    path = Path(path).resolve()
    prompt = path.read_text()
    for key, value in kwargs.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt

def query(prompt, input_image_paths=None, provider: Literal["openai", "gemini"]="openai", output: Literal["text", "json", "image"]="text"):
    input_image_paths = input_image_paths or []

    if provider == "openai":
        content = [{"type": "input_text", "text": prompt}, 
                   *[{"type": "input_image", "image_url": f"data:image/png;base64,{png_to_b64(path)}"} for path in input_image_paths]]
        kwargs = {"model": "gpt-5.6", "input": [{"role": "user", "content": content}]}
        if output == "json":
            kwargs["text"] = {"format": {"type": "json_object"}}
        elif output == "image":
            kwargs["tools"] = [{"type": "image_generation"}]

        response = openai_client.responses.create(**kwargs)
        if output == "image":
            for item in response.output:
                if item.type == "image_generation_call":
                    return base64.b64decode(item.result)
            raise RuntimeError("Model did not return an image")

        if output == "json":
            return json.loads(response.output_text)

        return response.output_text
    elif provider == "gemini":
        content = [{"type": "text", "text": prompt},
                   *[{"type": "image", "data": png_to_b64(path), "mime_type": "image/png"} for path in input_image_paths]]
        interaction = gemini_client.interactions.create(model=("gemini-3.1-flash-image" if output=="image" else "gemini-3.6-flash"), input=content)

        if output == "image":
            if interaction.output_image is None:
                raise RuntimeError("Model did not return an image")
            return base64.b64decode(interaction.output_image.data)

        if output == "json":
            return json.loads(interaction.output_text)

        return interaction.output_text

def edit_image(instruction, active_object, passive_object, input_image_path, output_image_path):
    prompt = load_prompt("prompts/edit_image.txt", instruction=instruction, active_object=active_object, passive_object=passive_object)
    png_bytes = query(prompt, [input_image_path], "gemini", "image")
    with open(output_image_path, "wb") as f:
        f.write(png_bytes)

    source = cv2.imread(str(input_image_path))
    edited = cv2.imread(str(output_image_path))
    edited = cv2.resize(edited, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_image_path, edited)

def get_sampling_constraints(instruction, active_object, passive_object, active_dimensions, translation_image_paths, rotation_image_paths, translation_output_path, rotation_output_path) -> SamplingConstraints:
    translation_prompt = load_prompt('prompts/get_sampling_constraints_translation.txt', instruction=instruction, active_object=active_object, passive_object=passive_object)
    rotation_prompt = load_prompt('prompts/get_sampling_constraints_rotation.txt', instruction=instruction, active_object=active_object, passive_object=passive_object)

    json_t = query(translation_prompt, translation_image_paths, "openai", "json")
    json_r = query(rotation_prompt, rotation_image_paths, "openai", "json")

    with open(translation_output_path, "w") as file:
        json.dump(json_t, file, indent=4)

    with open(rotation_output_path, "w") as file:
        json.dump(json_r, file, indent=4)

    max_active_dim = max(active_dimensions)
    S = SamplingConstraints(
        x=POSITION_CONSTRAINT_MAP[json_t["translation"]["x"]["assessment"]]*max_active_dim,
        y=POSITION_CONSTRAINT_MAP[json_t["translation"]["y"]["assessment"]]*max_active_dim,
        z=POSITION_CONSTRAINT_MAP[json_t["translation"]["z"]["assessment"]]*max_active_dim,
        x_bias=json_t["translation"]["x"]["direction"],
        y_bias=json_t["translation"]["y"]["direction"],
        z_bias=json_t["translation"]["z"]["direction"],
        roll=ORIENTATION_CONSTRAINT_MAP[json_r["rotation"]["roll"]["assessment"]],
        pitch=ORIENTATION_CONSTRAINT_MAP[json_r["rotation"]["pitch"]["assessment"]],
        yaw=ORIENTATION_CONSTRAINT_MAP[json_r["rotation"]["yaw"]["assessment"]],
    )
    return S

def get_validation_program(instruction, active_object, passive_object, active_dimensions, passive_dimensions, input_image_paths, output_path) -> ValidationProgram:
    prompt = load_prompt("prompts/get_validation_program.txt", instruction=instruction, active_object=active_object, passive_object=passive_object, 
                         active_dimensions=active_dimensions, passive_dimensions=passive_dimensions)
    json_raw = query(prompt, input_image_paths, "openai", "json")

    with open(output_path, "w") as file:
        json.dump(json_raw, file, indent=4)

    xyz_relations = [XYZRelation(axis=item["axis"], relation=item["relation"]) for item in json_raw.get("xyz_relations", [])]
    max_active_dim = max(active_dimensions)
    validation_mode = json_raw["validation_mode"]

    if validation_mode == "geometric_only":
        # Drift is not evaluated in geometric_only mode.
        x_drift = 0.0
        y_drift = 0.0
        z_drift = 0.0
        roll_drift = 0.0
        pitch_drift = 0.0
        yaw_drift = 0.0

    else:
        position = json_raw["position_drift_tolerance"]
        orientation = json_raw["orientation_drift_tolerance"]
        x_drift = (POSITION_TOLERANCE_MAP[position["x"]] * max_active_dim)
        y_drift = (POSITION_TOLERANCE_MAP[position["y"]] * max_active_dim)
        z_drift = (POSITION_TOLERANCE_MAP[position["z"]] * max_active_dim)

        # Current prompt calls rotation axes x/y/z.
        roll_drift = ORIENTATION_TOLERANCE_MAP[orientation["roll"]]
        pitch_drift = ORIENTATION_TOLERANCE_MAP[orientation["pitch"]]
        yaw_drift = ORIENTATION_TOLERANCE_MAP[orientation["yaw"]]

    return ValidationProgram(
        validation_mode=validation_mode,
        required_contacts=json_raw.get("required_contacts", []),
        forbidden_contacts=json_raw.get("forbidden_contacts", []),
        xyz_relations=xyz_relations,
        x_drift_tolerance=x_drift,
        y_drift_tolerance=y_drift,
        z_drift_tolerance=z_drift,
        roll_drift_tolerance=roll_drift,
        pitch_drift_tolerance=pitch_drift,
        yaw_drift_tolerance=yaw_drift,
    )

def select_best_candidate(instruction, active_object, passive_object, input_image_paths, output_path):
    prompt = load_prompt("prompts/select_best_candidate.txt", instruction=instruction, active_object=active_object, passive_object=passive_object)
    json_raw = query(prompt, input_image_paths, "openai", "json")
    with open(output_path, "w") as file:
        json.dump(json_raw, file, indent=4)

    return json_raw["best_index"]