import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import os
import time
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from scipy.spatial.transform import Rotation
from src.config import OBJECT_CATALOG, T_mujoco_board

POSITION_TOLERANCE_MAP = {"strict": 0.25,"moderate": 0.75, "permissive": 1.5} # multiply by the active object's longest dimension
ORIENTATION_TOLERANCE_MAP = {"strict": 7.5, "moderate": 30.0,"permissive": 60.0} 
POSITION_CONSTRAINT_MAP = {"correct": 0.25, "normal_search": 0.75, "major_revision": 1.5} # multiply by the active object's longest dimension
ORIENTATION_CONSTRAINT_MAP = {"correct": 15.0, "normal_search": 60.0, "major_revision": 120.0} 
REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = { # Approximated object meshes due to 3D-scanner being unable to produce fully accurate meshes 
    "small_dish": {
        "inertial_pos": (0, 0, 0.00418),
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.0034), "size": (0.035, 0.0034)},
            {"type": "cylinder", "pos": (0, 0, 0.0073), "size": (0.050, 0.0005)},
        ],
    },
    "big_dish": {
        "inertial_pos": (0, 0, 0.00450),
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.0037), "size": (0.0425, 0.0037)},
            {"type": "cylinder", "pos": (0, 0, 0.0077), "size": (0.0635, 0.0003)},
        ],
    },
    "small_cup": {
        "inertial_pos": (0, 0, 0.0636),
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.015), "size": (0.025, 0.015)},
            {"type": "box", "pos": (0, 0.0265, 0.069), "size": (0.028, 0.0015, 0.039)},
            {"type": "box", "pos": (0, -0.0265, 0.069), "size": (0.028, 0.0015, 0.039)},
            {"type": "box", "pos": (0.0265, 0, 0.069), "size": (0.0015, 0.025, 0.039)},
            {"type": "box", "pos": (-0.0265, 0, 0.069), "size": (0.0015, 0.025, 0.039)},
        ],
    },
    "cup": {
        "inertial_pos": (0, 0, 0.0636),
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.015), "size": (0.029, 0.015)},
            {"type": "box", "pos": (0, 0.0305, 0.069), "size": (0.032, 0.0015, 0.039)},
            {"type": "box", "pos": (0, -0.0305, 0.069), "size": (0.032, 0.0015, 0.039)},
            {"type": "box", "pos": (0.0305, 0, 0.069), "size": (0.0015, 0.029, 0.039)},
            {"type": "box", "pos": (-0.0305, 0, 0.069), "size": (0.0015, 0.029, 0.039)},
        ],
    },
    "large_cup": {
        "inertial_pos": (0, 0, 0.0636),
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.015), "size": (0.033, 0.015)},
            {"type": "box", "pos": (0, 0.0345, 0.069), "size": (0.036, 0.0015, 0.039)},
            {"type": "box", "pos": (0, -0.0345, 0.069), "size": (0.036, 0.0015, 0.039)},
            {"type": "box", "pos": (0.0345, 0, 0.069), "size": (0.0015, 0.033, 0.039)},
            {"type": "box", "pos": (-0.0345, 0, 0.069), "size": (0.0015, 0.033, 0.039)},
        ],
    },
}

@dataclass(frozen=True)
class XYZRelation: # passive is the reference, and active is the subject (e.g. "x greater_than" means "active object x greater than passive object x")
    axis: Literal["x", "y", "z"]
    relation: Literal["greater_than", "less_than"]

@dataclass(frozen=True)
class SamplingConstraints:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    x_bias: Literal["positive", "negative", "none"]
    y_bias: Literal["positive", "negative", "none"]
    z_bias: Literal["positive", "negative", "none"]
    def sample_dt(self, r, bias):
        if bias == "positive":
            return np.random.uniform(0, r)
        if bias == "negative":
            return np.random.uniform(-r, 0)
        return np.random.uniform(-r, r)
    
    def sample(self, T_goal, T_passive): # assumes T_goal and T_passive are in the same frame
        dx = self.sample_dt(self.x, self.x_bias)
        dy = self.sample_dt(self.y, self.y_bias)
        dz = self.sample_dt(self.z, self.z_bias)
        dr = np.random.uniform(-self.roll, self.roll)
        dp = np.random.uniform(-self.pitch, self.pitch)
        dyaw = np.random.uniform(-self.yaw, self.yaw)

        p_delta = T_passive[:3, :3] @ np.array([dx, dy, dz]) # apply translation in passive object's frame
        R_delta = Rotation.from_euler("xyz", [dr, dp, dyaw], degrees=True).as_matrix() # R_delta applied in object's own frame

        T_goal_new = T_goal.copy()
        T_goal_new[0:3, 3] += p_delta 
        T_goal_new[0:3, 0:3] = T_goal[0:3, 0:3] @ R_delta 

        return T_goal_new

@dataclass(frozen=True)
class ValidationProgram: 
    validation_mode: Literal["settling", "geometric_only"]
    required_contacts: list[Literal["active_passive", "active_table"]]
    forbidden_contacts: list[Literal["active_passive", "active_table"]]
    xyz_relations: list[XYZRelation]
    x_drift_tolerance: float
    y_drift_tolerance: float
    z_drift_tolerance: float
    roll_drift_tolerance: float
    pitch_drift_tolerance: float
    yaw_drift_tolerance: float
    max_initial_penetration: float = 0.05
    max_final_penetration: float = 0.0008

@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    active_body_id: int
    passive_body_id: int
    obstacle_body_ids: set[int]
    goal_mocap_id: int
    refined_goal_mocap_id: int
    table_geom_ids: set[int]
    xml_path: Path

class SceneBuilder:
    def __init__(self, lab_scene_path=REPO_ROOT / "models" / "lab_scene.xml"):
        self.lab_scene_path = Path(lab_scene_path).resolve()

    def build(self, active_name, passive_name, T_mujoco_active, T_mujoco_passive, T_mujoco_active_goal, obstacles, output_path) -> Scene: # OUTPUT_DIR = MODELS_DIR/"generated_xmls"
        root = ET.parse(self.lab_scene_path).getroot()
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
        compiler.set("meshdir", str(REPO_ROOT))

        asset = root.find("asset")
        worldbody = root.find("worldbody")
        if asset is None or worldbody is None:
            raise ValueError("[digital-twin]: lab_scene.xml misses asset or worldbody")
        for model_asset in asset.findall("model"):
            path = (self.lab_scene_path.parent / model_asset.get("file")).resolve()
            model_asset.set("file", os.path.relpath(path, output_path.parent))

        for name in dict.fromkeys((active_name, passive_name, *(name for name in obstacles))):
            mjcf = OBJECT_CATALOG[name]["mjcf"]
            if mjcf["type"] == "mesh":
                path = REPO_ROOT / mjcf["asset_xml"]
                relpath = Path(os.path.relpath(path, output_path.parent)).as_posix() # <include> locates file relative to the directory that contains the xml
                ET.SubElement(asset, "include", {"file": relpath})

        worldbody.append(self.make_body(active_name, T_mujoco_active, "active"))
        worldbody.append(self.make_body(passive_name, T_mujoco_passive, "passive"))
        worldbody.append(self.make_body(active_name, T_mujoco_active_goal, "goal"))
        worldbody.append(self.make_body(active_name, np.eye(4), "refined_goal"))
        for name, T_mujoco in obstacles:
            worldbody.append(self.make_body(name, T_mujoco, "obstacle"))

        ET.indent(root, space="  ")
        ET.ElementTree(root).write(output_path)

        model = mujoco.MjModel.from_xml_path(str(output_path))
        data = mujoco.MjData(model)
        active_body_id = model.body(f"active_{active_name}").id
        passive_body_id = model.body(f"passive_{passive_name}").id
        obstacle_body_ids = {model.body(f"obstacle_{name}").id for name, _ in obstacles}
        goal_mocap_id = int(model.body(f"goal_{active_name}").mocapid[0])
        refined_goal_mocap_id = int(model.body(f"refined_goal_{active_name}").mocapid[0])
        table_geom_ids = {model.geom(name).id for name in ("table", "table_base")}
        mujoco.mj_forward(model, data)
        return Scene(model, data, active_body_id, passive_body_id, obstacle_body_ids, goal_mocap_id, refined_goal_mocap_id, table_geom_ids, output_path)

    def make_body(self, name, T_mujoco, role):
        spec = OBJECT_CATALOG[name]
        mjcf = spec["mjcf"]
        physics = spec["physics"]
        body_name = f"{role}_{name}"

        # Build geometry
        if mjcf["type"] == "mesh":
            root = ET.parse(REPO_ROOT / mjcf["body_xml"]).getroot()
            body = root if root.tag == "body" else root.find(".//body")
            if body is None:
                raise ValueError(f"[digital-twin]: No <body> found in {mjcf['body_xml']}")

            for elem in body.iter():
                if elem is not body and elem.get("name"):
                    elem.set("name", f"{body_name}_{elem.get('name')}")

        elif mjcf["type"] == "primitive":
            primitive = PRIMITIVES[mjcf["primitive"]]
            z_offset = mjcf.get("z_offset", 0.0)
            body = ET.Element("body")
            for i, g in enumerate(primitive["geoms"]):
                pos = (*g["pos"][:2], g["pos"][2] + z_offset)
                ET.SubElement(body, "geom", {
                    "name": f"{body_name}_geom_{i}", "type": g["type"],
                    "pos": " ".join(map(str, pos)), "size": " ".join(map(str, g["size"]))
                })
        else:
            raise ValueError(f"[digital-twin]: Unknown MJCF type: {mjcf['type']}")

        # Set body name and pose
        q = Rotation.from_matrix(T_mujoco[:3, :3]).as_quat()
        body.attrib.clear()
        body.set("name", body_name)
        body.set("pos", " ".join(map(str, T_mujoco[:3, 3])))
        body.set("quat", f"{q[3]} {q[0]} {q[1]} {q[2]}")

        # Clean up existing body joints and geom attributes
        for parent in body.iter():
            for child in list(parent):
                if child.tag in {"joint", "freejoint"}:
                    parent.remove(child)

        for geom in body.iter("geom"):
            for attr in ("class", "mass", "friction", "contype", "conaffinity", "condim"):
                geom.attrib.pop(attr, None)

        ### Build bodies
        # Goal: visual-only mocap copy.
        if role in {"goal", "refined_goal"}:
            body.set("mocap", "true")

            inertial = body.find("inertial")
            if inertial is not None:
                body.remove(inertial)

            for geom in body.iter("geom"):
                geom.set("class", role)

            return body

        # Passive is fixed, so its inertial is unnecessary.
        if role in ("passive", "obstacle"):
            inertial = body.find("inertial")
            if inertial is not None:
                body.remove(inertial)

        # Active is free and needs inertia.
        if role == "active":
            body.insert(0, ET.Element(
                "freejoint",
                {"name": f"{body_name}_free"},
            ))

            mass = physics.get("mass")
            diaginertia = physics.get("diaginertia")

            if mjcf["type"] == "primitive":
                if mass is None or diaginertia is None:
                    raise ValueError(
                        f"[digital-twin]: {name} primitive requires mass and diaginertia"
                    )
                body.insert(0, ET.Element("inertial", {
                    "pos": " ".join(map(str, (*primitive["inertial_pos"][:2], primitive["inertial_pos"][2] + mjcf.get("z_offset", 0.0)))),
                    "mass": str(mass),
                    "diaginertia": " ".join(map(str, diaginertia)),
                }))
            else:
                inertial = body.find("inertial")
                if inertial is None:
                    raise ValueError(f"[digital-twin]: {name} body.xml has no <inertial>")
                if mass is not None:
                    inertial.set("mass", str(mass))
                if diaginertia is not None:
                    inertial.set("diaginertia", " ".join(map(str, diaginertia)))

        # Physical active/passive geoms get catalog contact parameters.
        for geom in body.iter("geom"):
            if physics.get("friction") is not None:
                geom.set("friction", " ".join(map(str, physics["friction"])))
            for attr in ("contype", "conaffinity", "condim"):
                if physics.get(attr) is not None:
                    geom.set(attr, str(physics[attr]))

        return body
                
class DigitalTwin:
    AXIS = {"x": 0, "y": 1, "z": 2}
    def __init__(self, scene: Scene, V: ValidationProgram, S: SamplingConstraints):
        self.scene = scene
        self.model = scene.model
        self.V = V
        self.S = S

        active_freejoint_id = self.model.body_jntadr[scene.active_body_id]
        self.active_qadr = self.model.jnt_qposadr[active_freejoint_id]
        self.active_dadr = self.model.jnt_dofadr[active_freejoint_id]

        self.T_goal = self.get_mocap_pose(scene.data, scene.goal_mocap_id)
        self.T_passive = self.get_body_pose(scene.data, scene.passive_body_id)
        self.viewer_data = None
        self.viewer = None


    def sample_and_validate(self, visualize=False, n_samples=1000, max_valid=10, sim_steps=500,
                        output_path=None, video_path=None):
        if visualize:
            import mujoco.viewer
            self.viewer_data = mujoco.MjData(self.model)
            with mujoco.viewer.launch_passive(self.model, self.viewer_data) as self.viewer:
                candidates = self.sample(n_samples, max_valid, sim_steps, True, video_path)
        else:
            candidates = self.sample(n_samples, max_valid, sim_steps, False, video_path)

        if output_path is not None:
            arr = np.stack(candidates) if candidates else np.empty((0, 4, 4))
            np.save(Path(output_path), arr)

        return candidates


    def sample(self, n_samples, max_valid, sim_steps, visualize=False, video_path=None):
        candidates = []
        renderer = video = render_data = camera = None

        if video_path is not None:
            w, h, fps, frames_per_sample = 640, 480, 30, 2

            render_data = mujoco.MjData(self.model)
            render_data.qpos[:] = self.scene.data.qpos
            render_data.qvel[:] = 0
            mujoco.mj_forward(self.model, render_data)

            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = render_data.xpos[self.scene.passive_body_id]
            camera.distance, camera.azimuth, camera.elevation = 0.55, 135.0, -25.0

            renderer = mujoco.Renderer(self.model, width=w, height=h)

            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not video.isOpened():
                renderer.close()
                raise RuntimeError(f"[digital-twin]: failed to open video writer: {video_path}")

        try:
            for i in range(n_samples):
                T_goal_sample = self.T_goal if i == 0 else self.S.sample(self.T_goal, self.T_passive)

                if visualize:
                    self.set_mocap_pose(self.viewer_data, self.scene.goal_mocap_id, T_goal_sample)
                    self.viewer.sync()
                    time.sleep(0.005)

                if video is not None:
                    self.set_mocap_pose(render_data, self.scene.goal_mocap_id, T_goal_sample)
                    renderer.update_scene(render_data, camera=camera)
                    bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                    for _ in range(frames_per_sample):
                        video.write(bgr)

                valid, T_final, reason = self.validate(T_goal_sample, sim_steps)
                if not valid:
                    continue

                candidates.append(T_final)
                print(f"[digital-twin]: valid candidate {len(candidates)}/{max_valid} discovered as the {i+1} sample")

                if len(candidates) >= max_valid:
                    break

        finally:
            if video is not None:
                video.release()
            if renderer is not None:
                renderer.close()

        if video_path is not None:
            print(f"[digital-twin]: sampling video saved to {video_path}")

        return candidates
                
    
    def validate(self, T_goal_sample, sim_steps):
        data = mujoco.MjData(self.model)
        self.set_body_pose(data, self.active_qadr, self.active_dadr, T_goal_sample)

        # Geometric-only tasks validate the proposed pose directly.
        if self.V.validation_mode == "geometric_only":
            for check in (
                lambda: self.check_penetration(data, "final"),
                lambda: self.check_forbidden_contact(data),
                lambda: self.check_required_contact(data),
                lambda: self.check_xyz_relations(data),
            ):
                valid, reason = check()
                if not valid: return False, T_goal_sample, reason
            return True, T_goal_sample, "valid"

        # Check initial pose before simulation.
        valid, reason = self.check_penetration(data, "initial")
        if not valid: return False, T_goal_sample, reason
        valid, reason = self.check_forbidden_contact(data)
        if not valid: return False, T_goal_sample, reason

        for _ in range(sim_steps):
            try: 
                mujoco.mj_step(self.model, data)
            except mujoco.FatalError as e:
                return False, T_goal_sample, f"mj_step failed: {e}"

            valid, reason = self.check_forbidden_contact(data)
            if not valid: return False, T_goal_sample, reason

        # Check final settled pose
        T_final = self.get_body_pose(data, self.scene.active_body_id)

        for check in (
            lambda: self.check_penetration(data, "final"),
            lambda: self.check_required_contact(data),
            lambda: self.check_drift(T_goal_sample, T_final),
            lambda: self.check_xyz_relations(data),
        ):
            valid, reason = check()
            if not valid: return False, T_final, reason

        return True, T_final, "valid"

    def check_penetration(self, data, type: Literal["initial", "final"]):
        max_penetration = 0.0

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body_id1, body_id2 = self.model.geom_bodyid[geom1], self.model.geom_bodyid[geom2]

            if body_id1 == self.scene.active_body_id or body_id2 == self.scene.active_body_id:
                max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))

        threshold = self.V.max_initial_penetration if type == "initial" else self.V.max_final_penetration
        if max_penetration > threshold:
            return False, f"{type} penetration {max_penetration} > {threshold}"
        return True, "valid"

    def check_forbidden_contact(self, data):
        contacts = set()
        scene_body_ids = {self.scene.active_body_id, self.scene.passive_body_id}
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body_id1, body_id2 = self.model.geom_bodyid[geom1], self.model.geom_bodyid[geom2]

            if {body_id1, body_id2} == scene_body_ids:
                contacts.add("active_passive")
            if ((body_id1 == self.scene.active_body_id and body_id2 in self.scene.obstacle_body_ids) or
               (body_id2 == self.scene.active_body_id and body_id1 in self.scene.obstacle_body_ids)):
               return False, "active object contacts obstacle"
            if ((body_id1 == self.scene.active_body_id and geom2 in self.scene.table_geom_ids) or
                (body_id2 == self.scene.active_body_id and geom1 in self.scene.table_geom_ids)):
                contacts.add("active_table")

        forbidden = set(self.V.forbidden_contacts) & contacts
        if forbidden:
            return False, f"forbidden contacts: {sorted(forbidden)}"
        return True, "valid"

    def check_required_contact(self, data):
        contacts = set()

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = self.model.geom_bodyid[geom1], self.model.geom_bodyid[geom2]

            if {body1, body2} == {self.scene.active_body_id, self.scene.passive_body_id}:
                contacts.add("active_passive")
            if ((body1 == self.scene.active_body_id and geom2 in self.scene.table_geom_ids) or
                (body2 == self.scene.active_body_id and geom1 in self.scene.table_geom_ids)):
                contacts.add("active_table")

        missing = set(self.V.required_contacts) - contacts
        if missing:
            return False, f"missing required contacts: {sorted(missing)}"
        return True, "valid"

    def check_drift(self, T_initial, T_final):
        position_drift = np.abs(T_final[:3, 3] - T_initial[:3, 3])
        position_limit = np.array([self.V.x_drift_tolerance, self.V.y_drift_tolerance, self.V.z_drift_tolerance])
        if np.any(position_drift > position_limit):
            return False, f"position drift {position_drift} > {position_limit}"

        R_delta = T_initial[:3, :3].T @ T_final[:3, :3]
        rotation_drift = np.abs(Rotation.from_matrix(R_delta).as_euler("xyz", degrees=True))
        rotation_limit = np.array([self.V.roll_drift_tolerance, self.V.pitch_drift_tolerance, self.V.yaw_drift_tolerance])
        if np.any(rotation_drift > rotation_limit):
            return False, f"rotation drift {rotation_drift} > {rotation_limit}"

        return True, "valid"

    def check_xyz_relations(self, data):
        p_world_active = data.xpos[self.scene.active_body_id]
        p_world_passive = data.xpos[self.scene.passive_body_id]
        R_world_passive = data.xmat[self.scene.passive_body_id].reshape(3, 3)

        p_passive_active = (
            R_world_passive.T @ (p_world_active - p_world_passive)
        )

        for relation in self.V.xyz_relations:
            a = p_passive_active[self.AXIS[relation.axis]]

            if relation.relation == "greater_than" and a <= 0:
                return False, f"active {relation.axis} <= passive origin"
            if relation.relation == "less_than" and a >= 0:
                return False, f"active {relation.axis} >= passive origin"

        return True, "valid"


    def get_body_pose(self, data, body_id):
        T = np.eye(4)
        T[:3,3] = data.xpos[body_id]
        T[:3,:3] = data.xmat[body_id].reshape(3,3)
        return T

    def get_mocap_pose(self, data, mocap_id):
        T = np.eye(4)
        q = data.mocap_quat[mocap_id]
        T[:3,3] = data.mocap_pos[mocap_id]
        T[:3,:3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return T

    def set_body_pose(self, data, body_qadr, body_dadr, T): # qaddr instead of body id because xpos is read-only 
        q = Rotation.from_matrix(T[:3,:3]).as_quat()
        data.qpos[body_qadr:body_qadr+7] = [*T[:3,3], q[3], q[0], q[1], q[2]]
        data.qvel[body_dadr:body_dadr+6] = 0
        mujoco.mj_forward(self.model, data)

    def set_mocap_pose(self, data, mocap_id, T):
        q = Rotation.from_matrix(T[:3,:3]).as_quat()
        data.mocap_pos[mocap_id] = T[:3,3]
        data.mocap_quat[mocap_id] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(self.model, data)