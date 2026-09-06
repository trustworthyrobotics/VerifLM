import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from trimesh.exchange.obj import export_obj
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from src.vlm import load_prompt, query


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

        # Save colored visual mesh as OBJ + MTL.
        mesh.visual = TextureVisuals(material=SimpleMaterial(diffuse=[*color, 255]))
        obj_text, files = export_obj(mesh, return_texture=True, mtl_name=f"{name}.mtl")
        mesh_path = mesh_dir / f"{name}.obj"
        mesh_path.write_text(obj_text)
        for filename, data in files.items():
            (mesh_dir / filename).write_bytes(data if isinstance(data, bytes) else data.encode())

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
        vis.create_window(width=640, height=640, visible=False)
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
        return np.percentile(pts, 75, axis=0) - np.percentile(pts, 25, axis=0)

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
        mesh.apply_scale(self._observed_extents(depth, mask, K).max() /
                         mesh.bounding_box.extents.max())

        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [90, 0, 180], degrees=True).as_matrix()
        mesh.apply_transform(T)

        color = np.median(rgb[mask], axis=0).astype(np.uint8)
        mesh, mesh_path, asset_xml, body_xml = self.processor.process(
            mesh, name, self.output_root, color
        )

        render_path = obj_dir / "physics_render.png"
        self.processor.render(mesh, color, render_path)

        dimensions = mesh.bounding_box.extents.astype(float)
        mass, friction = self._estimate_physics(prompt, dimensions, render_path)

        result = {
            "prompt": prompt,
            "mesh": str(mesh_path),
            "dimensions": tuple(map(float, dimensions)),
            "mjcf": {
                "type": "mesh",
                "asset_xml": str(asset_xml),
                "body_xml": str(body_xml),
            },
            "physics": {"mass": mass, "friction": friction},
        }
        (obj_dir / "estimate.json").write_text(json.dumps(result, indent=2))
        print(f"[object-estimation]: {name}: mass={mass:.3f} kg, friction={friction:.3f}")
        return result
