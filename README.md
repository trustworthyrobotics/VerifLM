# VerifLM

VerifLM is a refactored VLM-guided manipulation pipeline. It captures or loads an RGB-D source scene, edits a goal image with a VLM, segments active/passive objects, estimates object poses with FoundationPose, builds a MuJoCo digital twin, samples physically valid goal poses, asks a VLM to select the best candidate, and plans a pick/place trajectory.

## File Structure

```text
run_pipeline.py          CLI entrypoint
pipeline/
  main.py                pipeline orchestration and stage handling
  config.py              benchmark/task/object/motion configuration
  core.py                shared dataclasses, paths, transforms, serialization
  camera.py              RGB-D capture and calibration utilities
  vlm.py                 Gemini/OpenAI calls and prompt loading
  perception.py          VGGT depth, GroundingDINO/SAM2 masks, FoundationPose
  digital_twin.py        MuJoCo scene generation, constraints, sampling
  planner.py             grasp, IK, OMPL, and trajectory generation
  debug.py               interactive previews, mask review, debug artifacts
prompts/                 editable VLM prompt templates
models/                  MuJoCo XMLs, meshes, generated scene XMLs
extrinsics/              camera extrinsic calibration files
artifacts/               runtime outputs; ignored by git
```

## External Dependencies

Install the normal Python stack for this repo (`numpy`, `opencv-python`, `torch`, `mujoco`, `open3d`, `Pillow`, `python-dotenv`, `openai`, `google-genai`, etc.) plus these external systems, cloned as siblings of this repo (i.e. in the same parent directory as `VerifLM/`):

- **FoundationPose**: expected to expose `run_frame.py`; executed through a Docker container named by `FP_CONTAINER`.
- **VGGT**: uses VGGT (`facebook/VGGT-1B`) via `VGGT_REPO`.
- **GroundingDINO**: repo path must contain `groundingdino/config/GroundingDINO_SwinT_OGC.py` and `weights/groundingdino_swint_ogc.pth`.
- **SAM2**: repo path must contain `configs/sam2.1/sam2.1_hiera_l.yaml` and `checkpoints/sam2.1_hiera_large.pt`.
- **MuJoCo / OMPL / camera SDKs**: required for simulation, planning, and live RGB-D capture.

## Environment

Create `.env` in the repo root. Do not commit it.

```bash
# VLM providers
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Optional model overrides
OPENAI_VISION_MODEL=gpt-5.5
GEMINI_MODEL=gemini-2.0-flash
GEMINI_IMAGE_MODEL=gemini-2.0-flash-preview-image-generation
```

## Running

```bash
python run_pipeline.py --benchmark red_mug_on_plate --camera-type gemini
python run_pipeline.py --benchmark red_mug_on_plate --debug
python run_pipeline.py --benchmark red_mug_on_plate --skip-axis-reasoning
```

Sampling-axis reasoning is enabled by default; use `--skip-axis-reasoning` to fall back to configured sampling constraints.
