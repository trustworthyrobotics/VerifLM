# VerifLM

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
