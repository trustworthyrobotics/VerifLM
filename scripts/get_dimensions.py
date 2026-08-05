from pathlib import Path

import numpy as np
import trimesh
import sys
# Add ~/trl/VerifLM to Python's module search path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import OBJECT_CATALOG
from pipeline.core import model_path


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="scene", process=False)

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("scene contains no geometry")
        mesh = mesh.dump(concatenate=True)

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("mesh contains no vertices")

    return mesh


def main() -> None:
    print(f"{'object':<24} {'x (m)':>10} {'y (m)':>10} {'z (m)':>10}")
    print("-" * 58)

    for object_name, spec in OBJECT_CATALOG.items():
        mesh_relative_path = spec.get("mesh")

        if not mesh_relative_path:
            print(f"{object_name:<24} {'NO MESH':>32}")
            continue

        mesh_path = model_path(mesh_relative_path)

        try:
            mesh = load_mesh(mesh_path)

            # Axis-aligned dimensions in the mesh/object coordinate frame.
            dimensions = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
            print(
                f"{object_name:<24} "
                f"{dimensions[0]:>10.4f} "
                f"{dimensions[1]:>10.4f} "
                f"{dimensions[2]:>10.4f}"
            )

        except Exception as exc:
            print(f"{object_name:<24} ERROR: {exc}")


if __name__ == "__main__":
    main()