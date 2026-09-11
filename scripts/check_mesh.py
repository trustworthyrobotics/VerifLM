#!/usr/bin/env python3
"""Compare an estimated object's size with its ground-truth mesh."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from src.config import OBJECT_CATALOG


def load_mesh(path):
    mesh = trimesh.load(path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("object", help="OBJECT_CATALOG name, e.g. yellow_plate_small")
args = parser.parse_args()

root = Path(__file__).resolve().parent.parent
name = args.object

gt_path = Path(OBJECT_CATALOG[name]["mesh"])
if not gt_path.is_absolute():
    gt_path = root / gt_path

estimate_json = root / "artifacts" / "object_estimation" / name / "estimate.json"
estimated_path = Path(json.loads(estimate_json.read_text())["mesh"])
if not estimated_path.is_absolute():
    estimated_path = root / estimated_path

gt = load_mesh(gt_path)
estimated = load_mesh(estimated_path)

# Sort dimensions so an X/Y/Z permutation caused by mesh orientation does not
# look like a scale error.
gt_size = np.sort(gt.bounding_box.extents)
estimated_size = np.sort(estimated.bounding_box.extents)
ratios = estimated_size / gt_size

print(f"ground truth: {gt_path}")
print(f"estimated:    {estimated_path}")
print(f"GT size [m], sorted:        {gt_size}")
print(f"Estimated size [m], sorted: {estimated_size}")
print(f"Estimated / GT per axis:    {ratios}")
print(f"Longest-dimension ratio:    {estimated_size[-1] / gt_size[-1]:.4f}")

