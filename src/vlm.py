import base64
import json
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

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

def edit_image(instruction, active, passive, input_image_path, output_image_path):
    prompt = load_prompt("prompts/goal_image_edit.txt", instruction=instruction, active=active, passive=passive)
    png_bytes = query(prompt, [input_image_path], "gemini", "image")
    with open(output_image_path, "wb") as f:
        f.write(png_bytes)