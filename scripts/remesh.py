# Center all meshes at their geometric origins, then convex-decompose every mesh into a maximum of 256 convex hulls.
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import coacd
import trimesh


# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------

OBJECT_NAMES = [
    "wooden_rack",
    "wooden_stick_toy",
    "red_mug_small",
    "red_mug_big",
    "blue_mug_big",
    "green_plate_small",
    "pink_plate_small",
    "yellow_plate_small",
    "red_plate_big",
    "blue_plate_big",
    "teapot",
    "yellow_cap",
    "red_cup",
    "green_cup",
    "blue_cup",
]
COACD_THRESHOLD = 0.02
COACD_MAX_CONVEX_HULL = 256
COACD_RESOLUTION = 2000

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_MESH_DIR = REPO_ROOT / "models" / "meshes_old"
NEW_MESH_DIR = REPO_ROOT / "models" / "meshes"


# ---------------------------------------------------------------------
# Mesh processing
# ---------------------------------------------------------------------

def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load mesh: {path}")
    return mesh


def recenter_mesh(mesh):
    """Move the mesh's AABB center to the local-frame origin."""
    center = mesh.bounds.mean(axis=0)
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    return mesh, center


def decompose(mesh, name, output_dir):
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_CONVEX_HULL,
        resolution=COACD_RESOLUTION,
    )

    for i, (vertices, faces) in enumerate(parts):
        trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=True,
        ).export(output_dir / f"{name}_{i}.obj")

    return len(parts)


# ---------------------------------------------------------------------
# MJCF generation
# ---------------------------------------------------------------------

def write_asset_xml(name, n_parts, output_dir):
    root = ET.Element("asset")

    for i in range(n_parts):
        ET.SubElement(root, "mesh", {
            "name": f"{name}_{i}",
            "file": f"{name}_{i}.obj",
            "scale": "1 1 1",
        })

    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        output_dir / f"{name}_asset.xml",
        encoding="unicode",
    )


def write_body_xml(name, n_parts, output_dir):
    root = ET.Element("root")
    body = ET.SubElement(root, "body", {
        "name": name,
        "pos": "0 0 0",
        "quat": "1 0 0 0",
    })

    # SceneBuilder replaces mass and diaginertia with OBJECT_CATALOG values.
    ET.SubElement(body, "inertial", {
        "pos": "0 0 0",
        "mass": "1",
        "diaginertia": "1 1 1",
    })

    ET.SubElement(body, "freejoint", {
        "name": f"{name}_free",
    })

    for i in range(n_parts):
        ET.SubElement(body, "geom", {
            "class": name,
            "type": "mesh",
            "mesh": f"{name}_{i}",
        })

    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        output_dir / f"{name}_body.xml",
        encoding="unicode",
    )


# ---------------------------------------------------------------------
# Per-object processing
# ---------------------------------------------------------------------

def process_object(name):
    input_path = OLD_MESH_DIR / name / f"{name}.obj"
    object_dir = NEW_MESH_DIR / name
    remesh_dir = NEW_MESH_DIR / f"{name}_remeshed"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing mesh: {input_path}")

    # Remove old generated outputs so stale CoACD pieces cannot remain.
    if object_dir.exists():
        shutil.rmtree(object_dir)
    if remesh_dir.exists():
        shutil.rmtree(remesh_dir)

    object_dir.mkdir(parents=True)
    remesh_dir.mkdir(parents=True)

    print(f"\n[{name}] loading {input_path}")

    mesh = load_mesh(input_path)
    mesh, old_center = recenter_mesh(mesh)

    output_mesh = object_dir / f"{name}.obj"
    mesh.export(output_mesh)

    print(f"[{name}] old origin offset from center: {old_center}")
    print(f"[{name}] centered mesh -> {output_mesh}")

    n_parts = decompose(mesh, name, remesh_dir)

    write_asset_xml(name, n_parts, remesh_dir)
    write_body_xml(name, n_parts, remesh_dir)

    print(f"[{name}] CoACD -> {n_parts} convex parts")
    print(f"[{name}] assets -> {remesh_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    NEW_MESH_DIR.mkdir(parents=True, exist_ok=True)

    for name in OBJECT_NAMES:
        process_object(name)

    print("\nDone.")


if __name__ == "__main__":
    main()