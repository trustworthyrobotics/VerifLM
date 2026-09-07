import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from src.config import BENCHMARK_CONFIGS, OBJECT_CATALOG
from src.object_estimation import ObjectEstimator


def load_o3d(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    out.compute_vertex_normals()
    return mesh, out


def compare_meshes(true_path, pred_path):
    true_tm, true = load_o3d(true_path)
    pred_tm, pred = load_o3d(pred_path)

    # Predicted = solid red; true = green wireframe.
    pred.paint_uniform_color([0.8, 0.2, 0.2])
    true_lines = o3d.geometry.LineSet.create_from_triangle_mesh(true)
    true_lines.paint_uniform_color([0.1, 0.8, 0.1])

    ext_true = true_tm.bounding_box.extents
    ext_pred = pred_tm.bounding_box.extents
    print(f"\ntrue extents:      {np.round(ext_true, 4)} m")
    print(f"predicted extents: {np.round(ext_pred, 4)} m")
    print(f"pred/true ratio:   {np.round(ext_pred / ext_true, 3)}")

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * max(ext_true.max(), ext_pred.max())
    )
    o3d.visualization.draw_geometries(
        [pred, true_lines, axis],
        window_name="Predicted (red) vs True (green wireframe)",
    )


parser = argparse.ArgumentParser()
parser.add_argument("benchmark")
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--role", choices=("active", "passive"), default="active")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent
run_dir = ROOT / "artifacts" / f"{args.benchmark}_stage{args.stage}"
camera_dir = run_dir / "camera"
perception_dir = run_dir / "perception"

config = BENCHMARK_CONFIGS[args.benchmark]
name = config[args.role][args.stage]
catalog = OBJECT_CATALOG[name]

meta = json.loads((camera_dir / "source_camera_meta.json").read_text())
K = np.asarray(meta["K"], dtype=float)

result = ObjectEstimator(run_dir / "objects_test").estimate(
    name=name,
    prompt=catalog["prompt"],
    rgb_path=camera_dir / "source_rgb.png",
    depth_path=camera_dir / "source_depth_m.npy",
    mask_path=perception_dir / f"source_{args.role}_mask.png",
    K=K,
)

print(json.dumps(result, indent=2))
compare_meshes(catalog["mesh"], result["mesh"])