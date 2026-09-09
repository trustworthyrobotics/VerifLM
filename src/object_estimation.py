import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import cv2
import numpy as np
import open3d as o3d
import trimesh
import fast_simplification
from PIL import Image
from scipy.spatial.transform import Rotation
from trimesh.exchange.obj import export_obj
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals
from src.vlm import load_prompt, query


HUNYUAN_DIR = Path(__file__).resolve().parents[2] / "Hunyuan3D-2.1" / "hy3dshape"
sys.path.insert(0, str(HUNYUAN_DIR))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


class MeshProcessor:
    def __init__(self, threshold=0.02, max_hulls=256, resolution=2000):
        self.threshold, self.max_hulls, self.resolution = threshold, max_hulls, resolution

    def process(self, mesh, name, output_root, color):
        mesh_dir = Path(output_root) / name
        convex_dir = Path(output_root) / f"{name}_remeshed"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        if convex_dir.exists():
            shutil.rmtree(convex_dir)
        convex_dir.mkdir(parents=True)

        mesh = mesh.copy()
        mesh.apply_translation(-mesh.bounds.mean(axis=0))


        # Save as a vertex-colored OBJ.
        rgba = np.array([*color, 255], dtype=np.uint8)

        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))

        mesh_path = mesh_dir / f"{name}.obj"

        obj_text = export_obj(
            mesh,
            include_color=True,
            include_texture=False,
        )

        mesh_path.write_text(obj_text)

        # Convex collision decomposition.
        parts = coacd.run_coacd(
            coacd.Mesh(mesh.vertices, mesh.faces),
            threshold=self.threshold, max_convex_hull=self.max_hulls,
            resolution=self.resolution,
        )
        for i, (v, f) in enumerate(parts):
            trimesh.Trimesh(v, f, process=True).export(convex_dir / f"{name}_{i}.obj")

        asset = ET.Element("asset")
        for i in range(len(parts)):
            ET.SubElement(asset, "mesh", {"name": f"{name}_{i}", "file": f"{name}_{i}.obj"})
        ET.indent(asset, "  ")
        asset_path = convex_dir / f"{name}_asset.xml"
        ET.ElementTree(asset).write(asset_path, encoding="unicode")

        root = ET.Element("root")
        body = ET.SubElement(root, "body", {"name": name})
        ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
        rgba = " ".join(map(str, [*(np.asarray(color) / 255.0), 1.0]))
        for i in range(len(parts)):
            ET.SubElement(body, "geom", {"type": "mesh", "mesh": f"{name}_{i}", "rgba": rgba})
        ET.indent(root, "  ")
        body_path = convex_dir / f"{name}_body.xml"
        ET.ElementTree(root).write(body_path, encoding="unicode")

        return mesh, mesh_path, asset_path, body_path

    @staticmethod
    def render(mesh, color, output_path):
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
        o3d_mesh.paint_uniform_color(np.asarray(color) / 255.0)
        o3d_mesh.compute_vertex_normals()
        

        vis = o3d.visualization.Visualizer()
        if not vis.create_window(width=640, height=640, visible=True):
            raise RuntimeError("[object-estimation]: Open3D failed to create window")
        vis.add_geometry(o3d_mesh)
        vis.get_render_option().background_color = np.ones(3)
        vis.get_render_option().mesh_show_back_face = True
        ctr = vis.get_view_control()
        ctr.set_lookat(o3d_mesh.get_center())
        ctr.set_front([1, 1, 0.7])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_path), do_render=True)
        vis.destroy_window()


class ObjectEstimator:
    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.processor = MeshProcessor()
        self.pipeline = None

    def _get_pipeline(self):
        if self.pipeline is None:
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2.1"
            )
        return self.pipeline

    @staticmethod
    def _crop(rgb, mask, padding=30):
        ys, xs = np.where(mask)
        x0, x1 = max(0, xs.min()-padding), min(rgb.shape[1], xs.max()+padding+1)
        y0, y1 = max(0, ys.min()-padding), min(rgb.shape[0], ys.max()+padding+1)
        crop, m = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        return Image.fromarray(np.dstack((crop, m.astype(np.uint8)*255)), "RGBA")

    @staticmethod
    def _observed_extents(depth, mask, K):
        valid = mask & np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            raise ValueError("No valid depth inside mask")
        v, u = np.where(valid)
        z = depth[valid]
        pts = np.c_[(u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z]
        pts -= np.median(pts, axis=0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        pts = pts @ vt.T
        return np.percentile(pts, 99, axis=0) - np.percentile(pts, 1, axis=0)

    @staticmethod
    def _to_trimesh(mesh):
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.mesh_v),
            faces=np.asarray(mesh.mesh_f),
            process=False,
        )

    @staticmethod
    def _estimate_physics(prompt, dimensions, render_path):
        p = load_prompt(
            "prompts/estimate_physics.txt",
            object_name=prompt,
            dimensions=np.round(dimensions, 4).tolist(),
        )
        out = query(p, [render_path], provider="openai", output="json")
        mass, friction = float(out["mass"]), float(out["friction"])
        if mass <= 0 or friction < 0:
            raise ValueError(f"Invalid physics estimate: {out}")
        return mass, friction

    def load(self, name):
        return json.loads((self.output_root / name / "estimate.json").read_text())

    def estimate(self, name, prompt, rgb_path, depth_path, mask_path, K):
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        if not np.any(mask):
            raise ValueError(f"{name}: empty mask")

        obj_dir = self.output_root / name
        obj_dir.mkdir(parents=True, exist_ok=True)

        crop = self._crop(rgb, mask)
        crop.save(obj_dir / "observation.png")

        mesh = self._to_trimesh(self._get_pipeline()(image=crop)[0])

        raw_faces = len(mesh.faces)

        if raw_faces > 40_000:
            mesh = mesh.simplify_quadric_decimation(
                face_count=40_000,
                aggression=5,
            )
            print(f"[object-estimation]: simplified mesh {raw_faces:,} -> {len(mesh.faces):,} faces")
        observed = self._observed_extents(depth, mask, K)
        raw_extents = mesh.bounding_box.extents.copy()
        scale = observed.max() / raw_extents.max()

        print(
            f"[object-estimation]: {name} "
            f"depth=[{depth[mask].min():.3f}, {depth[mask].max():.3f}] m, "
            f"observed_extents={np.round(observed, 4)} m, "
            f"hunyuan_extents={np.round(raw_extents, 4)}, scale={scale:.4f}"
        )

        mesh.apply_scale(scale)
        mesh.apply_scale(self._observed_extents(depth, mask, K).max() / mesh.bounding_box.extents.max()) # new_mesh = old_mesh * (longest point cloud extent / bounding box extent)

        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
        mesh.apply_transform(T)

        color = np.median(rgb[mask], axis=0).astype(np.uint8)
        mesh, mesh_path, asset_xml, body_xml = self.processor.process(
            mesh, name, self.output_root, color
        )

        render_path = obj_dir / "physics_render.png"
        self.processor.render(mesh, color, render_path)

        dimensions = mesh.bounding_box.extents.astype(float)

        result = {
            "prompt": prompt,
            "mesh": str(mesh_path),
            "dimensions": tuple(map(float, dimensions)),
            "mjcf": {"type": "mesh", "asset_xml": str(asset_xml), "body_xml": str(body_xml)}
        }

        (obj_dir / "estimate.json").write_text(json.dumps(result, indent=2))
        return result