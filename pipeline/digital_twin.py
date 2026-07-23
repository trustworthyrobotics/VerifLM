from __future__ import annotations

import copy
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pipeline.config import OBJECT_CATALOG, VLM_PROVIDERS
from pipeline.core import (
    DigitalTwinResult, GoalRefinementResult, ObjectPose, PerceptionResult,
    REPO_ROOT, MODELS_DIR, SamplingConstraints, StagePaths, T_from_pos_quat,
    mat_to_quat_wxyz, matrix_to_rpy_deg, model_path, quat_wxyz_to_mat,
    rpy_to_quat,
)
from pipeline.vlm import (
    constraints_from_axis_reasoning, reason_sampling_axes_vlm,
    select_best_candidate_vlm,
)


def _body_transform(model, data, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    T = np.eye(4)
    T[:3, 3] = data.xpos[body_id]
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return T


# Scene XML generation
ROOT = REPO_ROOT
MODELS = MODELS_DIR


def generate_benchmark_xml(benchmark: str, config: dict) -> str:
    scene = config["scene"]
    active_name = scene["active_object"]
    passive_name = scene["passive_object"]
    candidate_names = list(scene.get("candidate_objects") or (active_name, passive_name))
    if active_name not in candidate_names:
        candidate_names.append(active_name)
    if passive_name not in candidate_names:
        candidate_names.append(passive_name)
    def _override_for(name: str) -> dict | None:
        if name == active_name:
            return scene.get("active_asset") or scene.get(f"{name}_asset")
        if name == passive_name:
            return scene.get("passive_asset") or scene.get(f"{name}_asset")
        return scene.get(f"{name}_asset")

    object_specs = {name: _object_spec(name, _override_for(name)) for name in candidate_names}
    active = object_specs[active_name]

    out_rel = scene.get("generated_xml_path", f"models/generated_xmls/{benchmark}.xml")
    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefix for include/model file paths: relative from the output XML back to models/
    _models_rel = Path(os.path.relpath(MODELS, out_path.parent)).as_posix()
    _models_prefix = "" if _models_rel == "." else f"{_models_rel}/"

    mujoco = ET.Element("mujoco", {"model": benchmark})
    repo_rel = Path(
        os.path.relpath(ROOT, out_path.parent)
    ).as_posix()

    ET.SubElement(
        mujoco,
        "compiler",
        {"meshdir": repo_rel},
    )
    _add_defaults(mujoco)
    ET.SubElement(mujoco, "size", {"nconmax": "1000", "njmax": "2000"})
    ET.SubElement(mujoco, "statistic", {"center": "0 0 0.9", "extent": "2.5"})
    ET.SubElement(
        mujoco,
        "option",
        {"gravity": "0 0 -9.81", "timestep": "0.002", "integrator": "implicitfast"},
    )
    _add_visual(mujoco)
    _add_assets(mujoco, tuple(object_specs.values()), _models_prefix)

    worldbody = ET.SubElement(mujoco, "worldbody")
    _add_shared_world(worldbody, surface_height=float(scene.get("surface_height", 0.7893)))
    object_poses = scene.get("object_poses") or {}
    default_object_poses = scene.get("default_object_poses") or {}

    def _stage_pose(name: str, default_pose: dict | None = None) -> dict | None:
        if name in object_poses:
            return object_poses[name]
        return default_pose or default_object_poses.get(name)

    for object_name in candidate_names:
        if object_name == active_name:
            continue
        default_pose = scene.get("passive_source_pose") if object_name == passive_name else None
        pose = _stage_pose(object_name, default_pose)
        if pose is None:
            raise ValueError(f"No source pose configured for static object {object_name!r}.")
        _append_object_body(worldbody, object_name, object_specs[object_name], pose, role="static")

    active_pose = _stage_pose(active_name, scene.get("active_source_pose"))
    if active_pose is None:
        raise ValueError(f"No source pose configured for active object {active_name!r}.")
    _append_object_body(worldbody, active_name, active, active_pose, role="active")
    _append_object_body(worldbody, active_name, active, scene.get("active_goal_pose"), role="goal")

    ET.indent(mujoco, space="  ")
    tree = ET.ElementTree(mujoco)
    tree.write(out_path, encoding="unicode", xml_declaration=False)
    return out_rel




def _object_spec(name: str, override: dict | None = None) -> dict:
    spec = copy.deepcopy(_object_catalog().get(name, {}))
    if override:
        spec.update(override)
    if not spec:
        raise ValueError(f"Unknown MuJoCo object {name!r}; add it to OBJECT_CATALOG or scene asset config.")
    return spec


def _object_catalog() -> dict:
    return OBJECT_CATALOG


def _add_defaults(mujoco: ET.Element) -> None:
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(
        ET.SubElement(default, "default", {"class": "goal"}),
        "geom",
        {"rgba": "0 1 0 0.3", "conaffinity": "0", "contype": "0"},
    )
    for name, spec in _object_catalog().items():
        if spec.get("primitive"):
            continue
        geom_attrs = {
            "friction": spec.get("geom_friction", " 0.01 0.001"),
            "condim": spec.get("geom_condim", "6"),
            "solref": spec.get("geom_solref", "0.002 1"),
            "solimp": spec.get("geom_solimp", "0.98 0.995 0.0001 0.5 2"),
        }
        ET.SubElement(
            ET.SubElement(default, "default", {"class": name}),
            "geom",
            geom_attrs,
        )


def _add_visual(mujoco: ET.Element) -> None:
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "-120", "elevation": "-20"})


def _add_assets(mujoco: ET.Element, objects: tuple[dict, ...], models_prefix: str = "") -> None:
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    # ET.SubElement(asset, "model", {"name": "kuka_iiwa_model", "file": f"{models_prefix}kuka_iiwa_14/iiwa14.xml"})
    ET.SubElement(asset, "model", {"name": "kuka_iiwa_model", "file": f"{models_prefix}kuka_iiwa_7/iiwa7.xml"})
    seen = set()
    for obj in objects:
        asset_xml = obj.get("asset_xml")
        if asset_xml and asset_xml not in seen:
            ET.SubElement(asset, "include", {"file": f"{models_prefix}{asset_xml}"})
            seen.add(asset_xml)


def _add_shared_world(worldbody: ET.Element, surface_height: float = 0.7893) -> None:
    table_half_height = surface_height / 2.0
    ET.SubElement(worldbody, "light", {"name": "main_light", "pos": "0 0 2.5", "dir": "0 0 -1"})
    ET.SubElement(worldbody, "geom", {"name": "ground", "type": "plane", "size": "3 3 0.1", "material": "groundplane"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table",
            "type": "box",
            "pos": f"0 0 {_fmt((table_half_height,))}",
            "size": f"0.50 0.75 {_fmt((table_half_height,))}",
            "rgba": "0.75 0.75 0.75 1",
            "friction": "0.05 0.001 0.0001",
            "condim": "4",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "arm1_plate", "type": "box", "pos": "0.40 0.65 0.75645", "size": "0.10 0.10 0.00645", "rgba": "0.3 0.3 0.3 1"},
    )
    arm1 = ET.SubElement(worldbody, "body", {"name": "arm1_mount", "pos": "0.40 0.65 0.7629", "euler": "0 0 180"})
    ET.SubElement(arm1, "attach", {"model": "kuka_iiwa_model", "body": "base", "prefix": "arm1_"})
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "arm2_plate", "type": "box", "pos": "0.40 -0.65 0.75645", "size": "0.10 0.10 0.00645", "rgba": "0.3 0.3 0.3 1"},
    )
    arm2 = ET.SubElement(worldbody, "body", {"name": "arm2_mount", "pos": "0.40 -0.65 0.7629", "euler": "0 0 180"})
    ET.SubElement(arm2, "attach", {"model": "kuka_iiwa_model", "body": "base", "prefix": "arm2_"})
    ET.SubElement(worldbody, "camera", {"name": "ceiling_cam", "mode": "fixed", "pos": "0 0 3.0", "xyaxes": "1 0 0  0 1 0"})


def _append_object_body(
    worldbody: ET.Element,
    name: str,
    spec: dict,
    pose: dict | None,
    role: str,
) -> None:
    if role not in {"active", "static", "goal"}:
        raise ValueError(f"Unsupported object role: {role!r}")
    if pose is None:
        if role == "goal":
            pose = {"xyz": (0.0, -2.0, 0.5), "rpy_deg": (0.0, 0.0, 0.0)}
        else:
            return
    if spec.get("primitive") in ("dish", "small_dish", "big_dish"):
        worldbody.append(
            _make_dish_body(name, pose, role, spec.get("primitive", "small_dish"))
        )
        return
    if spec.get("primitive") in ("cup", "small_cup", "large_cup"):
        worldbody.append(_make_cup_body(name, pose, role, spec.get("primitive", "cup")))
        return

    body_xml = spec.get("body_xml")
    if role == "goal":
        body_xml = spec.get("goal_body_xml") or _derive_goal_body_xml(body_xml)
    if not body_xml:
        if role == "goal":
            return
        raise ValueError(f"Object {name!r} does not define body_xml.")
    body = _load_body_xml(MODELS / body_xml)
    body_name = name if role != "goal" else f"{name}_goal"
    body.set("name", body_name)
    for child in body.iter():
        if child.tag == "freejoint":
            child.set("name", f"{body_name}_free")
        if role == "goal" and child.tag == "geom":
            child.set("class", "goal")
            child.set("contype", "0")
            child.set("conaffinity", "0")
    _set_body_pose(body, pose)
    if role == "static":
        _strip_joints(body)
    worldbody.append(body)


def _derive_goal_body_xml(body_xml: str | None) -> str | None:
    if not body_xml:
        return None
    candidate = str(body_xml).replace("_body.xml", "_goal_body.xml")
    return candidate if (MODELS / candidate).exists() else None

def _load_body_xml(path: Path) -> ET.Element:
    tree = ET.parse(path)
    body = tree.getroot().find("body")
    if body is None:
        raise ValueError(f"No <body> found in {path}")
    return copy.deepcopy(body)


def _strip_joints(body: ET.Element) -> None:
    """Make an included object body static by removing direct and nested joints."""
    for parent in body.iter():
        for child in list(parent):
            if child.tag in {"joint", "freejoint"}:
                parent.remove(child)



_PRIMITIVE_SIZES = {
    # (base_radius, base_half_height, rim_radius, rim_half_height)
    "small_dish": (0.035,  0.0034,  0.05,  0.0005),  # diameter 108mm, rim width 15mm, rim height 5mm
    "big_dish":   (0.0425, 0.0037, 0.0635, 0.0003),  # lower r=42.5mm (85/2) upper r=63.5mm (127/2)
    "small_cup":  (0.028,  0.041,  0.032,  0.013),   # body r=28mm h=82mm, mouth r=32mm h=26mm
    "cup":        (0.032,  0.041,  0.036,  0.013),   # body r=32mm h=82mm, mouth r=36mm h=26mm
    "large_cup":  (0.036,  0.041,  0.040,  0.013),   # body r=36mm h=82mm, mouth r=40mm h=26mm
}

_PRIMITIVE_MASSES = {
    "small_dish": (2.0,  0.5),
    "big_dish":   (2.0,  0.5),
    "small_cup":  (0.25, 0.05),
    "cup":        (0.25, 0.05),
    "large_cup":  (0.25, 0.05),
}


def _make_dish_body(
    name: str,
    pose: dict,
    role: str,
    primitive: str = "small_dish",
) -> ET.Element:
    br, bh, rr, rh = _PRIMITIVE_SIZES.get(
        primitive, _PRIMITIVE_SIZES["small_dish"]
    )
    body_mass, rim_mass = _PRIMITIVE_MASSES.get(
        primitive, _PRIMITIVE_MASSES["small_dish"]
    )
    suffix = "" if role != "goal" else "_goal"
    body = ET.Element("body", {"name": f"{name}{suffix}"})
    _set_body_pose(body, pose)
    if role in {"active", "goal"}:
        ET.SubElement(body, "freejoint", {"name": f"{name}{suffix}_free"})

    def _add_geom(gtype, pos, size, tag, mass):
        attrs = {
            "type": gtype,
            "name": f"{name}{suffix}_{tag}",
            "pos": _fmt(pos),
            "size": _fmt(size),
        }
        if role == "goal":
            attrs.update({"class": "goal", "contype": "0", "conaffinity": "0"})
        else:
            attrs.update({
                "condim": "4",
                "friction": "0.05 0.001 0.0001",
                "mass": f"{mass:.8g}",
            })
        ET.SubElement(body, "geom", attrs)

    _add_geom("cylinder", (0, 0, bh), (br, bh), "geom", body_mass)
    _add_geom(
        "cylinder",
        (0, 0, 2 * bh + rh),
        (rr, rh),
        "rim_geom",
        rim_mass,
    )
    return body


_CUP_WALL_T = 0.003   # 3 mm wall thickness for hollow cup primitives
_CUP_FILL_H = 0.030   # 30 mm solid fill at base; above this height the cup is hollow


def _make_cup_body(
    name: str,
    pose: dict,
    role: str,
    primitive: str = "cup",
) -> ET.Element:
    br, bh, _, rh = _PRIMITIVE_SIZES[primitive]
    body_mass, _ = _PRIMITIVE_MASSES[primitive]
    wall_t = _CUP_WALL_T
    fill_h = _CUP_FILL_H
    half_height = bh + rh

    suffix = "" if role != "goal" else "_goal"
    body = ET.Element("body", {"name": f"{name}{suffix}"})
    _set_body_pose(body, pose)
    if role in {"active", "goal"}:
        ET.SubElement(body, "freejoint", {"name": f"{name}{suffix}_free"})

    def _add_geom(gtype, pos, size, tag, mass):
        attrs = {
            "type": gtype,
            "name": f"{name}{suffix}_{tag}",
            "pos": _fmt(pos),
            "size": _fmt(size),
        }
        if role == "goal":
            attrs.update({"class": "goal", "contype": "0", "conaffinity": "0"})
        else:
            attrs.update({
                "condim": "4",
                "friction": "0.05 0.001 0.0001",
                "mass": f"{mass:.8g}",
            })
        ET.SubElement(body, "geom", attrs)

    _add_geom(
        "cylinder",
        (0, 0, fill_h / 2),
        (br - wall_t, fill_h / 2),
        "bottom",
        body_mass * 0.1,
    )

    wall_cz = half_height + fill_h / 2
    wall_hz = half_height - fill_h / 2
    panel_mass = body_mass * 0.225
    _add_geom("box", (0, br - wall_t / 2, wall_cz), (br, wall_t / 2, wall_hz), "wall_n", panel_mass)
    _add_geom("box", (0, -(br - wall_t / 2), wall_cz), (br, wall_t / 2, wall_hz), "wall_s", panel_mass)
    _add_geom("box", (br - wall_t / 2, 0, wall_cz), (wall_t / 2, br - wall_t, wall_hz), "wall_e", panel_mass)
    _add_geom("box", (-(br - wall_t / 2), 0, wall_cz), (wall_t / 2, br - wall_t, wall_hz), "wall_w", panel_mass)
    return body


def _set_body_pose(body: ET.Element, pose: dict) -> None:
    body.set("pos", _fmt(pose.get("xyz", (0.0, 0.0, 0.0))))
    quat = pose.get("quat")
    if quat is None:
        quat = _rpy_deg_to_quat(pose.get("rpy_deg", (0.0, 0.0, 0.0)))
    body.set("quat", _fmt(quat))
    body.attrib.pop("euler", None)


def _rpy_deg_to_quat(rpy_deg) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)

# Constraint and candidate rendering
def _load_mesh(path: str | Path):
    import open3d as o3d
    import trimesh

    loaded = trimesh.load(Path(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Unable to load mesh: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(loaded.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(loaded.faces))
    try:
        colors = np.asarray(loaded.visual.to_color().vertex_colors)
        if len(colors) == len(loaded.vertices):
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
    except Exception:
        pass
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_vertex_normals()
    return mesh


def _render(geometries, out_path: Path, camera: dict) -> np.ndarray:
    import open3d as o3d

    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1280, height=720, visible=False)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    options = visualizer.get_render_option()
    options.background_color = np.array([0.05, 0.05, 0.08])
    options.mesh_show_back_face = True
    control = visualizer.get_view_control()
    control.set_lookat(list(camera.get("lookat", (0.0, 0.0, 0.0))))
    control.set_front(list(camera.get("front", (0.0, -0.8, 0.8))))
    control.set_up(list(camera.get("up", (0.0, 0.0, 1.0))))
    control.set_zoom(float(camera.get("zoom", 0.9)))
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(str(out_path), do_render=True)
    visualizer.destroy_window()
    image = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to render {out_path}")
    return image


def render_relation(
    *,
    active_mesh: str | Path,
    passive_mesh: str | Path,
    T_world_active: np.ndarray,
    T_world_passive: np.ndarray,
    out_path: str | Path,
    camera: dict,
    frame: str,
    axis_size: float = 0.15,
) -> np.ndarray:
    """Render active relative to passive, with either active or world axes."""
    import open3d as o3d

    relative = np.linalg.inv(T_world_passive) @ T_world_active
    passive = _load_mesh(passive_mesh)
    active = copy.deepcopy(_load_mesh(active_mesh))
    active.transform(relative)

    frame_T = relative if frame == "active" else np.eye(4)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    axes.transform(frame_T)
    image = _render([passive, active, axes], Path(out_path), camera)
    label = (
        "Active frame: X/roll red, Y/pitch green, Z/yaw blue"
        if frame == "active"
        else "World frame: X red, Y green, Z blue"
    )
    cv2.putText(image, label, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), image)
    return image


def render_constraint_views(
    *,
    active_mesh: str | Path,
    passive_mesh: str | Path,
    T_world_active: np.ndarray,
    T_world_passive: np.ndarray,
    views: tuple[dict, dict],
    out_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotation, translation = [], []
    for index, camera in enumerate(views):
        suffix = "" if index == 0 else "_reverse"
        rotation.append(
            render_relation(
                active_mesh=active_mesh,
                passive_mesh=passive_mesh,
                T_world_active=T_world_active,
                T_world_passive=T_world_passive,
                out_path=out_dir / f"goal_active_axes{suffix}.png",
                camera=camera,
                frame="active",
            )
        )
        translation.append(
            render_relation(
                active_mesh=active_mesh,
                passive_mesh=passive_mesh,
                T_world_active=T_world_active,
                T_world_passive=T_world_passive,
                out_path=out_dir / f"goal_world_axes{suffix}.png",
                camera=camera,
                frame="world",
            )
        )
    return rotation, translation

# Physics-based pose sampling
"""
Pose sampling and validation via physics simulation.
Extract from test_sampler.py with only essential functionality.
"""






@dataclass
class _SamplerState:
    benchmark: str = ""
    n_samples: int = 0
    max_candidate_images: int = 0
    normal_position_range: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal_orientation_range: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_variance_multipliers: dict = field(default_factory=dict)
    orientation_variance_multipliers: dict = field(default_factory=dict)
    initial_max_pen: float = 0.0
    final_max_pen: float = 0.0
    sim_steps: int = 0
    max_z_change: float = 0.0
    max_ori_change: float = 0.0
    contact_entities: dict = field(default_factory=dict)
    axis_variance_multipliers: dict = field(default_factory=lambda: {
        axis: 1.0 for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    })
    pos_bias: dict = field(default_factory=lambda: {
        axis: None for axis in ("x", "y", "z")
    })
    required_contacts: list = field(default_factory=list)
    forbidden_contacts: list = field(default_factory=lambda: [("active", "table")])
    min_active_z: float | None = None
    max_xy_offset_from_passive: float | None = None
    model: mujoco.MjModel | None = None
    eval_data: mujoco.MjData | None = None
    active_body_id: int | None = None
    active_qadr: int | None = None
    active_dadr: int | None = None
    passive_body_id: int = -1
    table_geom_id: int = -1

_state = _SamplerState()

def configure(config: dict, *, benchmark: str | None = None) -> None:
    """Load stage-specific sampling and contact settings."""
    sampling = config["sampling"]
    _state.benchmark = benchmark or config.get("benchmark", _state.benchmark)
    _state.n_samples = int(sampling["n_samples"])
    _state.max_candidate_images = int(sampling["max_candidate_images"])
    _state.normal_position_range = np.asarray(sampling["normal_position_range"], dtype=float).copy()
    _state.normal_orientation_range = np.deg2rad(
        np.asarray(sampling["normal_orientation_range"], dtype=float)
    )
    _state.position_variance_multipliers = dict(sampling["position_variance_multipliers"])
    _state.orientation_variance_multipliers = dict(sampling["orientation_variance_multipliers"])
    _state.initial_max_pen = float(sampling["initial_max_pen"])
    _state.final_max_pen = float(sampling["final_max_pen"])
    _state.sim_steps = int(sampling["sim_steps"])
    _state.max_z_change = float(sampling["max_z_change"])
    _state.max_ori_change = math.radians(float(sampling["max_ori_change"]))
    _state.contact_entities = dict(config["scene"]["contact_entities"])
    _state.required_contacts = [tuple(pair) for pair in sampling.get("required_contacts", [("active", "passive")])]
    _state.forbidden_contacts = [tuple(pair) for pair in sampling.get("forbidden_contacts", [("active", "table")])]
    _state.min_active_z = sampling.get("min_active_z")
    _state.max_xy_offset_from_passive = sampling.get("max_xy_offset_from_passive")
    _state.pos_bias = {axis: sampling.get("pos_bias", {}).get(axis) for axis in ("x", "y", "z")}



def set_normal_sampling_ranges(position_xyz=None, orientation_rpy_deg=None):
    """
    Override the normal sampling ranges for position and orientation.
    
    Args:
        position_xyz: [x, y, z] half-ranges in meters
        orientation_rpy_deg: [roll, pitch, yaw] half-ranges in degrees
    """
    if position_xyz is not None:
        _state.normal_position_range[:] = np.asarray(position_xyz)
    if orientation_rpy_deg is not None:
        _state.normal_orientation_range[:] = np.deg2rad(
            np.asarray(orientation_rpy_deg)
        )


def set_constraints(
    axis_variance_multipliers=None,
    pos_bias=None,
):
    """
    Set sampling space constraints based on VLM analysis.
    
    Args:
        axis_variance_multipliers: numeric sampling-range multiplier for each axis
        pos_bias: dict mapping axis to "positive", "negative", or None
    """
    if axis_variance_multipliers is not None:
        _state.axis_variance_multipliers.update(axis_variance_multipliers)
    if pos_bias is not None:
        _state.pos_bias.update(pos_bias)


def setup(mujoco_model, eval_data, active_body_name: str):
    """Initialize sampler with model, eval data, and active body name."""
    _state.model = mujoco_model
    _state.eval_data = eval_data
    
    # Get body id and qpos/qvel addresses for the active body
    _state.active_body_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_BODY, active_body_name
    )
    if _state.active_body_id < 0:
        raise ValueError(f"Body not found: {active_body_name}")
    
    jid = mujoco_model.body_jntadr[_state.active_body_id]
    if jid < 0 or mujoco_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body must have <freejoint/>: {active_body_name}")
    
    _state.active_qadr = mujoco_model.jnt_qposadr[jid]
    _state.active_dadr = mujoco_model.jnt_dofadr[jid]

    passive_name = _state.contact_entities.get("passive")
    _state.passive_body_id = (
        mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, passive_name)
        if passive_name else -1
    )
    table_name = _state.contact_entities.get("table", "table")
    _state.table_geom_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, table_name
    )


def quat_angle(q1, q2):
    """Smallest angle between two quaternions (in radians)."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * math.acos(dot)


def set_active_pose(d, pos, quat, zero_velocity=True):
    """Kinematically set active object pose."""
    d.qpos[_state.active_qadr:_state.active_qadr + 3] = pos
    d.qpos[_state.active_qadr + 3:_state.active_qadr + 7] = quat
    if zero_velocity:
        d.qvel[_state.active_dadr:_state.active_dadr + 6] = 0.0


def safe_forward(d, label="mj_forward"):
    """Run mj_forward safely without killing the script on pathological contacts."""
    try:
        mujoco.mj_forward(_state.model, d)
        return True
    except mujoco.FatalError as e:
        print(f"[reject] {label} failed: {e}")
        return False


def safe_step(d, label="mj_step"):
    """Run mj_step safely."""
    try:
        mujoco.mj_step(_state.model, d)
        return True
    except mujoco.FatalError as e:
        print(f"[reject] {label} failed: {e}")
        return False


# ----------------------------
# Contact helpers
# ----------------------------
def active_contact_info(d, active_only=True):
    """
    Return max penetration and contacts, optionally limited to the active body.
    
    MuJoCo contact.dist:
      dist < 0  => penetration
      dist = 0  => touching
      dist > 0  => near contact / within margin
    """
    if not safe_forward(d, "active_contact_info/mj_forward"):
        return float("inf"), []

    contacts = []
    max_pen = 0.0

    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = _state.model.geom_bodyid[g1]
        b2 = _state.model.geom_bodyid[g2]

        if active_only and b1 != _state.active_body_id and b2 != _state.active_body_id:
            continue

        geom1_name = mujoco.mj_id2name(_state.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        geom2_name = mujoco.mj_id2name(_state.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        body1_name = mujoco.mj_id2name(_state.model, mujoco.mjtObj.mjOBJ_BODY, b1)
        body2_name = mujoco.mj_id2name(_state.model, mujoco.mjtObj.mjOBJ_BODY, b2)

        dist = float(c.dist)
        pen = max(0.0, -dist)
        max_pen = max(max_pen, pen)

        contacts.append({
            "body1": body1_name,
            "geom1": geom1_name,
            "body2": body2_name,
            "geom2": geom2_name,
            "dist": dist,
            "penetration": pen,
        })

    return max_pen, contacts


def missing_required_contacts(contacts):
    """Return required active/passive/table pairs absent after settling."""
    missing = []
    for symbolic_a, symbolic_b in _state.required_contacts:
        entity_a = _state.contact_entities[symbolic_a]
        entity_b = _state.contact_entities[symbolic_b]
        found = any(
            (
                entity_a in (c["body1"], c["geom1"])
                and entity_b in (c["body2"], c["geom2"])
            )
            or (
                entity_b in (c["body1"], c["geom1"])
                and entity_a in (c["body2"], c["geom2"])
            )
            for c in contacts
        )
        if not found:
            missing.append((symbolic_a, symbolic_b))
    return missing


def _active_touches_table(d) -> bool:
    """Direct geom-ID check: is any geom of the active body in contact with the table?"""
    if _state.table_geom_id < 0:
        return False
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(_state.model.geom_bodyid[g1])
        b2 = int(_state.model.geom_bodyid[g2])
        if (g1 == _state.table_geom_id and b2 == _state.active_body_id) or \
           (g2 == _state.table_geom_id and b1 == _state.active_body_id):
            return True
    return False


def final_rejection_reasons(final_pen, z_change, ori_change, missing_contacts,
                            final_pos=None, passive_pos=None, forbidden_contacts_found=None):
    """Describe every final-state validation failure."""
    reasons = []
    if final_pen >= _state.final_max_pen:
        reasons.append(
            f"final penetration {final_pen:.6f} >= limit "
            f"{_state.final_max_pen:.6f}"
        )
    if z_change >= _state.max_z_change:
        reasons.append(
            f"vertical settling {z_change:.6f} m >= limit "
            f"{_state.max_z_change:.6f} m"
        )
    if ori_change >= _state.max_ori_change:
        reasons.append(
            f"orientation drift {math.degrees(ori_change):.2f} deg "
            f">= limit {math.degrees(_state.max_ori_change):.2f} deg"
        )
    if missing_contacts:
        reasons.append(f"missing required contacts: {missing_contacts}")
    if forbidden_contacts_found:
        reasons.append(f"forbidden contacts (active touches table): {forbidden_contacts_found}")
    if _state.min_active_z is not None and final_pos is not None:
        if float(final_pos[2]) < _state.min_active_z:
            reasons.append(
                f"active object Z {final_pos[2]:.4f} m < min_active_z {_state.min_active_z:.4f} m"
            )
    limit = _state.max_xy_offset_from_passive
    if limit is not None and final_pos is not None and passive_pos is not None:
        xy_offset = float(np.linalg.norm(
            np.array(final_pos[:2]) - np.array(passive_pos[:2])
        ))
        if xy_offset > limit:
            reasons.append(
                f"XY offset from passive {xy_offset:.4f} m > limit {limit:.4f} m"
            )
    return reasons


# ----------------------------
# Sampling / validation
# ----------------------------
def _candidate_failure(reason, initial_pen=float("inf")):
    return False, {
        "reason": reason,
        "initial_pen": initial_pen,
        "final_pen": np.nan,
        "z_change": np.nan,
        "ori_change": np.nan,
        "final_pos": None,
        "final_quat": None,
    }


def _restore_reference_state(reference_state, label):
    qpos, qvel = reference_state
    _state.eval_data.qpos[:] = qpos
    _state.eval_data.qvel[:] = qvel
    safe_forward(_state.eval_data, label)


def evaluate_candidate(pos, quat, reference_state):
    """
    Validate a single candidate pose.
    
    Args:
        pos: position [3]
        quat: quaternion wxyz [4]
        reference_state: (qpos, qvel) tuple to initialize eval_data from
    
    Returns:
        (valid, info) where valid is bool and info is dict with metrics
    """
    qpos0, qvel0 = reference_state
    _state.eval_data.qpos[:] = qpos0
    _state.eval_data.qvel[:] = qvel0

    set_active_pose(_state.eval_data, pos, quat, zero_velocity=True)
    if not safe_forward(_state.eval_data, "candidate initial mj_forward"):
        _restore_reference_state(
            reference_state,
            "candidate restore after initial forward fail",
        )
        return _candidate_failure("initial mj_forward failed")

    initial_pen, _ = active_contact_info(_state.eval_data)
    if initial_pen >= _state.initial_max_pen:
        _restore_reference_state(
            reference_state,
            "candidate restore after initial reject",
        )
        return _candidate_failure("initial penetration too large", initial_pen)

    init_z = float(pos[2])
    init_quat = quat.copy()

    # Kick with a small random horizontal angular velocity to destabilize any
    # unstable equilibrium (e.g. plate balanced on its rim). The kick is in the
    # XZ plane (perpendicular to gravity) so it always acts as a tipping torque
    # regardless of the object's orientation. Magnitude 0.05 rad/s: small enough
    # that a properly-constrained pose stays within max_ori_change, large enough
    # that an inverted-pendulum-like unstable balance topples within sim_steps.
    _kick_angle = np.random.uniform(0.0, 2.0 * np.pi)
    _state.eval_data.qvel[_state.active_dadr + 3:_state.active_dadr + 6] += np.array(
        [np.cos(_kick_angle), 0.0, np.sin(_kick_angle)], dtype=float
    ) * 0.05

    for i in range(_state.sim_steps):
        if not safe_step(_state.eval_data, f"candidate mj_step {i}"):
            _restore_reference_state(
                reference_state,
                "candidate restore after mj_step fail",
            )
            return _candidate_failure(
                f"mj_step crashed at step {i}",
                initial_pen,
            )

    if not safe_forward(_state.eval_data, "candidate final mj_forward"):
        _restore_reference_state(
            reference_state,
            "candidate restore after final forward fail",
        )
        return _candidate_failure("final mj_forward failed", initial_pen)

    final_pos = _state.eval_data.qpos[
        _state.active_qadr:_state.active_qadr + 3
    ].copy()
    final_quat = _state.eval_data.qpos[
        _state.active_qadr + 3:_state.active_qadr + 7
    ].copy()

    final_pen, _ = active_contact_info(_state.eval_data)
    _, final_contacts = active_contact_info(
        _state.eval_data, active_only=False
    )
    z_change = abs(float(final_pos[2]) - init_z)
    ori_change = quat_angle(init_quat, final_quat)
    missing_contacts = missing_required_contacts(final_contacts)
    forbidden_contacts_found = [("active", "table")] if _active_touches_table(_state.eval_data) else []

    passive_pos = (
        _state.eval_data.xpos[_state.passive_body_id].copy()
        if _state.passive_body_id >= 0 else None
    )
    rejection_reasons = final_rejection_reasons(
        final_pen,
        z_change,
        ori_change,
        missing_contacts,
        final_pos=final_pos,
        passive_pos=passive_pos,
        forbidden_contacts_found=forbidden_contacts_found,
    )
    valid = not rejection_reasons
    reason = "valid" if valid else "; ".join(rejection_reasons)

    info = {
        "reason": reason,
        "initial_pen": initial_pen,
        "final_pen": final_pen,
        "z_change": z_change,
        "ori_change": ori_change,
        "missing_required_contacts": missing_contacts,
        "final_pos": final_pos,
        "final_quat": final_quat,
    }

    return valid, info


def sample_and_validate_poses(
    base_pos,
    base_quat,
    n_samples=None,
    max_valid_candidates=None,
    verbose=True,
    stop_after_valid_limit=True,
    sample_callback=None,
    pause_on_valid=False,
    pause_callback=None,
):
    """Sample nearby poses and keep the candidates that pass physics checks."""
    sample_count = _state.n_samples if n_samples is None else int(n_samples)
    candidate_limit = (
        _state.max_candidate_images
        if max_valid_candidates is None
        else int(max_valid_candidates)
    )
    if candidate_limit <= 0:
        raise ValueError("max_valid_candidates must be positive")

    valid_candidates = []
    base_rotation = quat_wxyz_to_mat(base_quat)
    reference_state = (
        _state.eval_data.qpos.copy(),
        _state.eval_data.qvel.copy(),
    )

    if verbose:
        print(f"[sampler] sampling {sample_count} candidates")

    limit_reported = False
    for index in range(sample_count):
        local_pos_offset = np.zeros(3, dtype=float)
        for axis_index, axis in enumerate(("x", "y", "z")):
            axis_range = (
                _state.normal_position_range[axis_index]
                * _state.axis_variance_multipliers[axis]
            )
            if _state.pos_bias[axis] == "positive":
                low, high = 0.0, abs(axis_range)
            elif _state.pos_bias[axis] == "negative":
                low, high = -abs(axis_range), 0.0
            else:
                low, high = -axis_range, axis_range
            local_pos_offset[axis_index] = np.random.uniform(low, high)
        sample_pos = base_pos + local_pos_offset

        orientation_range = _state.normal_orientation_range.copy()
        for axis_index, axis in enumerate(("roll", "pitch", "yaw")):
            orientation_range[axis_index] *= _state.axis_variance_multipliers[axis]
        delta_rpy = np.random.uniform(-orientation_range, orientation_range)
        sample_quat = mat_to_quat_wxyz(
            base_rotation @ quat_wxyz_to_mat(rpy_to_quat(*delta_rpy))
        )

        if sample_callback is not None:
            sample_callback(sample_pos, sample_quat)

        valid, info = evaluate_candidate(sample_pos, sample_quat, reference_state)
        if verbose and index % 50 == 0:
            print(
                f"  {index:3d}/{sample_count} valid={valid:5} | "
                f"reason={info['reason']:<35} | "
                f"initial_pen={info['initial_pen']:.4f} | "
                f"final_pen={info['final_pen']:.4f}"
            )

        if valid:
            if len(valid_candidates) < candidate_limit:
                valid_candidates.append((info["final_pos"], info["final_quat"], info))
                if pause_on_valid or pause_callback is not None:
                    if sample_callback is not None:
                        sample_callback(info["final_pos"], info["final_quat"])
                    if pause_callback is not None:
                        pause_callback(len(valid_candidates), info)
                    else:
                        input(f"  [pause] valid candidate #{len(valid_candidates)} - press Enter to continue")
            if len(valid_candidates) >= candidate_limit:
                if verbose and not limit_reported:
                    print(
                        f"[sampler] reached valid candidate limit "
                        f"({candidate_limit})"
                    )
                limit_reported = True
                if stop_after_valid_limit:
                    break

    if verbose:
        print(f"[sampler] found {len(valid_candidates)} valid candidate(s)")
    return valid_candidates


# ----------------------------
# Pipeline helpers: constrained sampling screenshots + attached-object OMPL
# ----------------------------

def apply_sampling_constraints(constraints) -> None:
    """Apply VLM sampling constraints or an equivalent dict."""
    if constraints is None:
        return
    if isinstance(constraints, dict):
        levels = constraints.get(
            "sampling_variance",
            constraints.get("noise_factors", {}),
        )
        bias = constraints.get("pos_bias", constraints.get("directional_bias", {}))
        contact_target = constraints.get("contact_target", "passive")
    else:
        levels = constraints.sampling_variance
        bias = constraints.pos_bias
        contact_target = constraints.contact_target
    axis_variance_multipliers = {
        axis: _state.position_variance_multipliers.get(
            levels.get(axis, "medium"),
            _state.position_variance_multipliers["medium"],
        )
        for axis in ("x", "y", "z")
    }
    axis_variance_multipliers.update({
        axis: _state.orientation_variance_multipliers.get(
            levels.get(axis, "medium"),
            _state.orientation_variance_multipliers["medium"],
        )
        for axis in ("roll", "pitch", "yaw")
    })
    contact_target = str(contact_target).strip().lower()
    if contact_target not in {"passive", "table"}:
        contact_target = "passive"
    _state.required_contacts = [("active", contact_target)]
    set_constraints(
        axis_variance_multipliers=axis_variance_multipliers,
        pos_bias=bias,
    )


def _goal_freejoint_qadr(model, goal_freejoint_name: str | None) -> int:
    if not goal_freejoint_name:
        return -1
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, goal_freejoint_name)
    return int(model.jnt_qposadr[joint_id]) if joint_id >= 0 else -1


def _set_goal_pose(data, goal_qadr: int, pos, quat) -> None:
    if goal_qadr < 0:
        return
    data.qpos[goal_qadr:goal_qadr + 3] = pos
    data.qpos[goal_qadr + 3:goal_qadr + 7] = quat


def _render_goal_candidate_image(pos, quat, out_path, goal_qadr: int, width=640, height=480, camera_name=None):
    """Render the MuJoCo scene with the visual goal body at the candidate pose."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qpos0 = _state.eval_data.qpos.copy()
    qvel0 = _state.eval_data.qvel.copy()
    _set_goal_pose(_state.eval_data, goal_qadr, pos, quat)
    safe_forward(_state.eval_data, "candidate render forward")

    renderer = None
    try:
        renderer = mujoco.Renderer(_state.model, height=height, width=width)
        if camera_name:
            renderer.update_scene(_state.eval_data, camera=camera_name)
        else:
            renderer.update_scene(_state.eval_data)
        rgb = renderer.render()
    finally:
        if renderer is not None:
            renderer.close()
        _state.eval_data.qpos[:] = qpos0
        _state.eval_data.qvel[:] = qvel0
        safe_forward(_state.eval_data, "candidate render restore")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, out_path.stem, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), bgr)
    return str(out_path)


def sample_validate_and_render_candidates(
    base_pos,
    base_quat,
    *,
    out_dir,
    n_samples=None,
    max_images=None,
    camera_name=None,
    goal_freejoint_name: str | None = None,
    verbose=True,
    stop_after_valid_limit=True,
    sample_callback=None,
    pause_on_valid=False,
    pause_callback=None,
):
    """Validate sampled poses and render valid candidates from the MuJoCo scene."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_limit = _state.max_candidate_images if max_images is None else int(max_images)
    if image_limit <= 0:
        raise ValueError("max_images must be positive")
    for old_candidate in out_dir.glob("candidate_*.png"):
        old_candidate.unlink()

    candidates = sample_and_validate_poses(
        base_pos,
        base_quat,
        n_samples=n_samples,
        max_valid_candidates=image_limit,
        verbose=verbose,
        stop_after_valid_limit=stop_after_valid_limit,
        sample_callback=sample_callback,
        pause_on_valid=pause_on_valid,
        pause_callback=pause_callback,
    )

    goal_qadr = _goal_freejoint_qadr(_state.model, goal_freejoint_name)
    if goal_qadr < 0:
        raise RuntimeError(f"Goal freejoint not found in MuJoCo model: {goal_freejoint_name!r}")
    for index, (pos, quat, info) in enumerate(candidates):
        image_path = out_dir / f"candidate_{index:03d}.png"
        info["image_path"] = _render_goal_candidate_image(
            pos,
            quat,
            image_path,
            goal_qadr=goal_qadr,
            camera_name=camera_name,
        )
    return candidates


# Digital-twin alignment
def _settle_active(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    active_body: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    *,
    max_steps: int = 600,
    min_steps: int = 30,
    velocity_tolerance: float = 5e-4,
) -> np.ndarray:
    active_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, active_body)
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}
    dof_width = {0: 6, 1: 3, 2: 1, 3: 1}
    frozen_qpos, frozen_dofs, active_dofs = [], [], []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        qadr, dadr = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
        nq, nd = qpos_width[joint_type], dof_width[joint_type]
        target = active_dofs if model.jnt_bodyid[joint_id] == active_id else frozen_dofs
        target.extend(range(dadr, dadr + nd))
        if model.jnt_bodyid[joint_id] != active_id:
            frozen_qpos.extend(range(qadr, qadr + nq))

    set_active_pose(data, position, quaternion, zero_velocity=True)
    mujoco.mj_forward(model, data)
    frozen_values = data.qpos[frozen_qpos].copy()
    for step in range(max_steps):
        mujoco.mj_step(model, data)
        data.qpos[frozen_qpos] = frozen_values
        data.qvel[frozen_dofs] = 0.0
        if step >= min_steps and np.linalg.norm(data.qvel[active_dofs]) < velocity_tolerance:
            break
    mujoco.mj_forward(model, data)
    return _body_transform(model, data, active_body)


class DigitalTwin:
    """Owns MuJoCo scene alignment, physical settling, and goal-pose sampling."""

    def __init__(self, args, benchmark_config: dict, perception: PerceptionResult, paths: StagePaths):
        self.args = args
        self.config = benchmark_config
        self.perception = perception
        self.paths = paths
        self.result = self._build()

    def _build(self) -> DigitalTwinResult:
        scene = self.config["scene"]
        passive_name = scene["contact_entities"]["passive"]
        table_height = scene.get("surface_height")
        tag_z_mj = float(os.getenv("TAG_Z_MJ", "0.7893"))
        if table_height is not None:
            tag_z_mj = float(table_height + self.perception.source_passive.T_world_object[2, 3])

        T_board_to_mj = np.array(
            [
                [1.0, 0.0, 0.0, self.args.board_x_mj],
                [0.0, -1.0, 0.0, self.args.board_y_mj],
                [0.0, 0.0, -1.0, tag_z_mj],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T_passive = T_board_to_mj @ self.perception.source_passive.T_world_object
        if table_height is not None:
            T_passive[2, 3] = float(table_height)
        T_goal = T_passive @ self.perception.relative_goal

        generated = copy.deepcopy(self.config)
        passive_pose = {
            "xyz": tuple(float(v) for v in T_passive[:3, 3]),
            "rpy_deg": matrix_to_rpy_deg(T_passive[:3, :3]),
        }
        generated["scene"]["passive_source_pose"] = passive_pose
        generated["scene"]["default_object_poses"][self.args.stage["passive"]] = passive_pose
        relative_xml = generate_benchmark_xml(self.args.benchmark, generated)
        xml_path = model_path(relative_xml)

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        configure(self.config, benchmark=self.args.benchmark)
        setup(model, data, self.args.mujoco_active_body)
        set_normal_sampling_ranges(
            position_xyz=self.args.normal_position_range,
            orientation_rpy_deg=self.args.normal_orientation_range_deg,
        )

        actual_passive = _body_transform(model, data, passive_name)
        T_goal[:3, 3] += actual_passive[:3, 3] - T_passive[:3, 3]
        T_passive = actual_passive

        source_initial = T_board_to_mj @ self.perception.source_active.T_world_object
        source_initial[2, 3] = max(source_initial[2, 3], tag_z_mj)
        if self.args.skip_settlement:
            source_settled = source_initial
            set_active_pose(data, source_initial[:3, 3], mat_to_quat_wxyz(source_initial[:3, :3]))
            mujoco.mj_forward(model, data)
        else:
            source_settled = _settle_active(
                model,
                data,
                self.args.mujoco_active_body,
                source_initial[:3, 3],
                mat_to_quat_wxyz(source_initial[:3, :3]),
            )

        return DigitalTwinResult(
            xml_path=xml_path,
            model=model,
            data=data,
            source_pose_mj=ObjectPose(
                self.perception.source_active.name,
                source_settled,
                self.perception.source_active.confidence,
                "FoundationPose followed by MuJoCo settlement",
            ),
            passive_pose_mj=ObjectPose(passive_name, T_passive, 1.0, "MuJoCo scene pose"),
            initial_goal_pose_mj=ObjectPose(
                self.perception.goal_active.name,
                T_goal,
                self.perception.goal_active.confidence,
                "Edited active/passive relation anchored to the real passive object",
            ),
        )

    def sample(self, constraints: SamplingConstraints, debug=None) -> list[tuple[np.ndarray, np.ndarray, dict]]:
        if self.args.skip_sampling:
            return []
        apply_sampling_constraints(constraints)
        if getattr(debug, "enabled", False):
            return self._sample_with_debug_viewer()

        goal = self.result.initial_goal_pose_mj.T_world_object
        goal_freejoint_name = self.config["scene"].get("goal_freejoint_name")
        return sample_validate_and_render_candidates(
            goal[:3, 3],
            mat_to_quat_wxyz(goal[:3, :3]),
            out_dir=self.paths.candidates,
            n_samples=self.args.n_samples,
            max_images=self.args.max_candidate_images,
            goal_freejoint_name=goal_freejoint_name,
            verbose=True,
        )

    def _sample_with_debug_viewer(self) -> list[tuple[np.ndarray, np.ndarray, dict]]:
        try:
            import mujoco.viewer as mj_viewer
            from mujoco.glfw import glfw
        except ImportError:
            print("[debug] mujoco.viewer is unavailable; sampling without interactive pauses")
            goal = self.result.initial_goal_pose_mj.T_world_object
            return sample_validate_and_render_candidates(
                goal[:3, 3],
                mat_to_quat_wxyz(goal[:3, :3]),
                out_dir=self.paths.candidates,
                n_samples=self.args.n_samples,
                max_images=self.args.max_candidate_images,
                goal_freejoint_name=self.config["scene"].get("goal_freejoint_name"),
                verbose=True,
            )

        model, data = self.result.model, self.result.data
        goal = self.result.initial_goal_pose_mj.T_world_object
        goal_freejoint_name = self.config["scene"].get("goal_freejoint_name")
        goal_qadr = _goal_freejoint_qadr(model, goal_freejoint_name)
        if goal_qadr < 0:
            raise RuntimeError(f"Goal freejoint not found in MuJoCo model: {goal_freejoint_name!r}")

        _set_goal_pose(data, goal_qadr, goal[:3, 3], mat_to_quat_wxyz(goal[:3, :3]))
        mujoco.mj_forward(model, data)

        start_sampling = [False]
        valid_continue = [False]

        def key_callback(keycode: int) -> None:
            if keycode in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                start_sampling[0] = True
                valid_continue[0] = True

        def show_goal_pose(pos, quat) -> None:
            _set_goal_pose(data, goal_qadr, pos, quat)
            mujoco.mj_forward(model, data)
            viewer.sync()

        def pause_on_valid(candidate_number: int, info=None) -> None:
            valid_continue[0] = False
            print(
                f"  [pause] valid candidate #{candidate_number} - "
                "press Enter in the MuJoCo window to continue",
                flush=True,
            )
            while not valid_continue[0] and viewer.is_running():
                viewer.sync()
                time.sleep(0.016)

        with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            print("[debug] Press Enter in the MuJoCo window to start sampling", flush=True)
            while not start_sampling[0] and viewer.is_running():
                viewer.sync()
                time.sleep(0.016)
            if not viewer.is_running():
                return []

            return sample_validate_and_render_candidates(
                goal[:3, 3],
                mat_to_quat_wxyz(goal[:3, :3]),
                out_dir=self.paths.candidates,
                n_samples=self.args.n_samples,
                max_images=self.args.max_candidate_images,
                goal_freejoint_name=goal_freejoint_name,
                verbose=True,
                sample_callback=show_goal_pose,
                pause_on_valid=True,
                pause_callback=pause_on_valid,
            )

    def pose_from_candidate(self, candidate, note: str) -> ObjectPose:
        position, quaternion, _ = candidate
        return ObjectPose(
            self.perception.goal_active.name,
            T_from_pos_quat(position, quaternion),
            confidence=0.5,
            note=note,
        )

# Goal refinement facade
class GoalRefinementPipeline:
    """Convert the perceived goal into a physically validated MuJoCo goal pose."""

    def run(
        self,
        *,
        args,
        benchmark_config: dict,
        perception: PerceptionResult,
        active_prompt: str,
        passive_prompt: str,
        paths: StagePaths,
        debug=None,
        progress=None,
    ) -> GoalRefinementResult:
        constraints = self._constraints(
            args,
            benchmark_config,
            perception,
            active_prompt,
            passive_prompt,
            paths,
            progress=progress,
        )
        if progress is not None:
            progress.step(7, "building MuJoCo digital twin")
        twin = DigitalTwin(args, benchmark_config, perception, paths)
        if progress is not None:
            progress.step(8, "sampling valid active-object goal candidates in MuJoCo")
        candidates = twin.sample(constraints, debug=debug)
        rendered = [candidate for candidate in candidates if candidate[2].get("image_path")]

        if rendered:
            if progress is not None:
                progress.step(9, "selecting best physically valid candidate with VLM")
            selection = select_best_candidate_vlm(
                [candidate[2]["image_path"] for candidate in rendered],
                instruction=args.instruction,
                active_object=active_prompt,
                passive_object=passive_prompt,
                provider=VLM_PROVIDERS.get("step_9", "gemini"),
            )
            refined_goal = twin.pose_from_candidate(
                rendered[selection["best_index"]],
                f"VLM-selected physically valid candidate: {selection.get('reasoning', '')}",
            )
        else:
            if progress is not None:
                progress.step(9, "using transferred goal pose; no valid sampled candidates")
            selection = {
                "best_index": -1,
                "reasoning": "No valid sampled candidate; using the transferred initial goal pose.",
            }
            refined_goal = twin.result.initial_goal_pose_mj

        (paths.outputs / "candidate_selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        return GoalRefinementResult(twin, refined_goal, constraints, selection)

    @staticmethod
    def _constraints(
        args,
        benchmark_config: dict,
        perception: PerceptionResult,
        active_prompt: str,
        passive_prompt: str,
        paths: StagePaths,
        progress=None,
    ) -> SamplingConstraints:
        if progress is not None:
            progress.step(
                6,
                (
                    "rendering 3D scenes and reasoning about sampling axes with VLM"
                    if args.axis_reasoning
                    else "rendering 3D scenes and using configured sampling constraints"
                ),
            )
        corrected_goal_passive = perception.goal_passive.T_world_object.copy()
        corrected_goal_passive[:3, :3] = perception.source_passive.T_world_object[:3, :3]
        rotation_views, translation_views = render_constraint_views(
            active_mesh=model_path(args.active_mesh),
            passive_mesh=model_path(args.passive_mesh),
            T_world_active=perception.goal_active.T_world_object,
            T_world_passive=corrected_goal_passive,
            views=benchmark_config["views"],
            out_dir=paths.inputs,
        )
        if not args.axis_reasoning:
            configured_bias = benchmark_config["sampling"].get("pos_bias", {})
            return SamplingConstraints(
                sampling_variance={axis: "medium" for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
                pos_bias={axis: configured_bias.get(axis) for axis in ("x", "y", "z")},
                contact_target="passive",
                reasoning="VLM axis reasoning disabled by runtime configuration.",
            )

        reasoning = reason_sampling_axes_vlm(
            rotation_views,
            translation_views,
            args.instruction,
            active_prompt,
            passive_prompt,
            provider=VLM_PROVIDERS.get("step_6", "openai"),
        )
        (paths.outputs / "sampling_axis_reasoning.json").write_text(
            json.dumps(reasoning, indent=2), encoding="utf-8"
        )
        return constraints_from_axis_reasoning(reasoning)
