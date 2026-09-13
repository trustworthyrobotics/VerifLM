# VerifLM

VerifLM generates language-conditioned 6D object goal poses, verifies them in MuJoCo, and plans collision-free robot trajectories. It supports predefined object models and mesh-free operation using reconstructed geometry, estimated physics, and parallel-world validation.

## Installation

VerifLM targets Python 3.11 with CUDA-capable PyTorch.

```bash
python -m pip install -r requirements.txt
```

The OMPL Python bindings must be importable as:

```python
from ompl import base, geometric
```

Use `opencv-contrib-python`, not `opencv-python`, because camera calibration requires `cv2.aruco`.

## External Dependencies

Clone these repositories as siblings of `VerifLM/`:

```text
workspace/
├── VerifLM/
├── FoundationPose/
├── GroundingDINO/
├── sam2/
├── VGGT/
└── Hunyuan3D-2.1/
```

Expected resources:

* **FoundationPose:** `FoundationPose/run_frame.py`, executed in the Docker container `foundationpose_realsense`.
* **VGGT:** the `vggt` package must be importable; `facebook/VGGT-1B` weights are loaded automatically.
* **GroundingDINO:** requires `GroundingDINO_SwinT_OGC.py` and `weights/groundingdino_swint_ogc.pth`.
* **SAM2:** requires `sam2.1_hiera_l.yaml` and `checkpoints/sam2.1_hiera_large.pt`.
* **Hunyuan3D 2.1:** required for mesh-free object reconstruction.
* **MuJoCo, OMPL, Docker, and Intel RealSense SDK:** required for simulation, planning, pose estimation, * * **AnyGrasp**:  expected under `anygrasp_sdk/grasp_detection/`.

The main pipeline currently uses catalog grasps by default.

## Environment

Create `.env` in the repository root:

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Do not commit this file.

## Usage

Run commands from the repository root.

### Predefined meshes

```bash
python -m src.main mug_on_plate
```

### Mesh-free input

```bash
python -m src.main mug_on_plate --estimate-objects --clear-gpu
```

Reconstructed objects are cached under `artifacts/object_estimation/`.

## Reusing Results

Available flags include:

* `--reuse-extrinsics`
* `--reuse-source`
* `--reuse-edited`
* `--reuse-perception`
* `--reuse-VS`
* `--skip-candidate-selection`
* `--clear-gpu`
