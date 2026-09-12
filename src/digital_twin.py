import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import os
import copy
import time
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from scipy.spatial.transform import Rotation

POSITION_TOLERANCE_MAP = {"strict": 0.25,"moderate": 0.75, "permissive": 1.5} # multiply by the active object's longest dimension
ORIENTATION_TOLERANCE_MAP = {"strict": 20.0, "moderate": 40.0,"permissive": 60.0} 
POSITION_CONSTRAINT_MAP = {"correct": 0.25, "normal_search": 0.75, "major_revision": 1.5} # multiply by the active object's longest dimension
ORIENTATION_CONSTRAINT_MAP = {"correct": 20.0, "normal_search": 60.0, "major_revision": 180.0} 
REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = { # Approximated object meshes due to 3D-scanner being unable to produce fully accurate meshes 
    "small_dish": {
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.0034), "size": (0.035, 0.0034)},
            {"type": "cylinder", "pos": (0, 0, 0.0073), "size": (0.050, 0.0005)},
        ],
    },
    "big_dish": {
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.0037), "size": (0.0425, 0.0037)},
            {"type": "cylinder", "pos": (0, 0, 0.0077), "size": (0.0635, 0.0003)},
        ],
    },
    "small_cup": {
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.015), "size": (0.025, 0.015)},
            {"type": "box", "pos": (0, 0.0265, 0.069), "size": (0.028, 0.0015, 0.039)},
            {"type": "box", "pos": (0, -0.0265, 0.069), "size": (0.028, 0.0015, 0.039)},
            {"type": "box", "pos": (0.0265, 0, 0.069), "size": (0.0015, 0.025, 0.039)},
            {"type": "box", "pos": (-0.0265, 0, 0.069), "size": (0.0015, 0.025, 0.039)},
        ],
    },
    "cup": {
        "geoms": [
            {"type": "cylinder", "pos": (0, 0, 0.015), "size": (0.029, 0.015)},
            {"type": "box", "pos": (0, 0.0305, 0.069), "size": (0.032, 0.0015, 0.039)},
            {"type": "box", "pos": (0, -0.0305, 0.069), "size": (0.032, 0.0015, 0.039)},
            {"type": "box", "pos": (0.0305, 0, 0.069), "size": (0.0015, 0.029, 0.039)},
            {"type": "box", "pos": (-0.0305, 0, 0.069), "size": (0.0015, 0.029, 0.039)},
        ],
    },
    "large_cup": {
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
    
    def sample(self, T_goal, T_passive, world_aligned=False): # assumes poses share one world frame
        dx = self.sample_dt(self.x, self.x_bias)
        dy = self.sample_dt(self.y, self.y_bias)
        dz = self.sample_dt(self.z, self.z_bias)
        dr = np.random.uniform(-self.roll, self.roll)
        dp = np.random.uniform(-self.pitch, self.pitch)
        dyaw = np.random.uniform(-self.yaw, self.yaw)

        delta = np.array([dx, dy, dz])
        p_delta = delta if world_aligned else T_passive[:3, :3] @ delta
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
    max_initial_penetration: float = 0.015
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
    def __init__(self, object_catalog, lab_scene_path=REPO_ROOT / "models" / "lab_scene.xml"):
        self.object_catalog = object_catalog
        self.lab_scene_path = Path(lab_scene_path).resolve()

    @staticmethod
    def _resolve(path):
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path

    def build(self, active_name, passive_name, T_mujoco_active, T_mujoco_passive,
              T_mujoco_active_goal, obstacles, output_path, overrides=None) -> Scene:
        if overrides is not None:
            catalog = copy.deepcopy(self.object_catalog)
            for name, values in overrides.items():
                catalog[name].update(values)
            return SceneBuilder(catalog, self.lab_scene_path).build(active_name, passive_name, T_mujoco_active,
                T_mujoco_passive, T_mujoco_active_goal, obstacles, output_path)
        output_path = Path(output_path)
        root = ET.parse(self.lab_scene_path).getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)
        compiler.set("meshdir", str(REPO_ROOT))
        compiler.set("inertiafromgeom", "true")

        asset, worldbody = root.find("asset"), root.find("worldbody")
        if asset is None or worldbody is None:
            raise ValueError("[digital-twin]: lab_scene.xml misses asset or worldbody")

        for model_asset in asset.findall("model"):
            path = (self.lab_scene_path.parent / model_asset.get("file")).resolve()
            model_asset.set("file", os.path.relpath(path, output_path.parent))

        names = dict.fromkeys((active_name, passive_name, *(name for name, _ in obstacles)))
        for name in names:
            mjcf = self.object_catalog[name]["mjcf"]
            if mjcf["type"] == "mesh":
                path = self._resolve(mjcf["asset_xml"])
                scale = float(self.object_catalog[name].get("mesh_scale", 1.0))
                if scale == 1.0:
                    ET.SubElement(asset, "include", {"file": Path(os.path.relpath(path, output_path.parent)).as_posix()})
                else:
                    if not np.isfinite(scale) or scale <= 0: raise ValueError("Invalid mesh scale")
                    for elem in ET.parse(path).getroot():
                        elem = copy.deepcopy(elem)
                        if elem.tag == "mesh":
                            elem.set("file", str((path.parent / elem.get("file")).resolve()))
                            base = np.fromstring(elem.get("scale", "1 1 1"), sep=" ")
                            elem.set("scale", " ".join(map(str, base * scale)))
                        asset.append(elem)

        active_body = self.make_body(active_name, T_mujoco_active, "active")
        worldbody.append(active_body)
        worldbody.append(self.make_body(passive_name, T_mujoco_passive, "passive"))
        worldbody.append(self.make_body(active_name, T_mujoco_active_goal, "goal"))
        worldbody.append(self.make_body(active_name, np.eye(4), "refined_goal"))
        for name, T_mujoco in obstacles:
            worldbody.append(self.make_body(name, T_mujoco, "obstacle"))

        # Probe with density=1, then use one uniform density that gives the target mass.
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(output_path)
        probe = mujoco.MjModel.from_xml_path(str(output_path))
        active_body_id = probe.body(f"active_{active_name}").id
        volume = float(probe.body_mass[active_body_id])
        mass = float(self.object_catalog[active_name]["physics"]["mass"])
        if volume <= 0 or mass <= 0:
            raise ValueError(f"[digital-twin]: invalid inferred volume/mass for {active_name}")

        density = mass / volume
        for geom in active_body.iter("geom"):
            geom.set("density", str(density))
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

        print(f"[digital-twin]: {active_name}: mass={model.body_mass[active_body_id]:.3f} kg, "
              f"density={density:.1f} kg/m^3")
        return Scene(model, data, active_body_id, passive_body_id, obstacle_body_ids,
                     goal_mocap_id, refined_goal_mocap_id, table_geom_ids, output_path)

    def make_body(self, name, T_mujoco, role):
        spec = self.object_catalog[name]
        mjcf, physics = spec["mjcf"], spec["physics"]
        body_name = f"{role}_{name}"

        if mjcf["type"] == "mesh":
            root = ET.parse(self._resolve(mjcf["body_xml"])).getroot()
            body = root if root.tag == "body" else root.find(".//body")
            if body is None:
                raise ValueError(f"[digital-twin]: no <body> in {mjcf['body_xml']}")
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
            raise ValueError(f"[digital-twin]: unknown MJCF type: {mjcf['type']}")

        q = Rotation.from_matrix(T_mujoco[:3, :3]).as_quat()
        body.attrib.clear()
        body.set("name", body_name)
        body.set("pos", " ".join(map(str, T_mujoco[:3, 3])))
        body.set("quat", f"{q[3]} {q[0]} {q[1]} {q[2]}")

        for parent in body.iter():
            for child in list(parent):
                if child.tag in {"joint", "freejoint", "inertial"}:
                    parent.remove(child)
        for geom in body.iter("geom"):
            for attr in ("class", "mass", "density", "friction", "contype", "conaffinity", "condim"):
                geom.attrib.pop(attr, None)

        if role in {"goal", "refined_goal"}:
            body.set("mocap", "true")
            for geom in body.iter("geom"):
                geom.set("class", role)
            return body

        if role == "active":
            body.insert(0, ET.Element("freejoint", {"name": f"{body_name}_free"}))
            for geom in body.iter("geom"):
                geom.set("density", "1")

        mu = float(physics["friction"])
        for geom in body.iter("geom"):
            geom.set("friction", f"{mu} 0 0")
            geom.set("condim", "3")

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


    def rank_ensemble(self, worlds, n_samples=1000, max_candidates=5, sim_steps=250, output_path=None):
        poses, masks = [], []
        for i in range(n_samples):
            pose = self.T_goal.copy() if i == 0 else self.S.sample(self.T_goal, self.T_passive)
            poses.append(pose.copy())
            masks.append([world.validate(pose.copy(), sim_steps)[0] for world in worlds])
            if (i + 1) % 10 == 0: print(f"[ensemble]: scored {i+1}/{n_samples} proposals")
        poses = np.asarray(poses).reshape(-1, 4, 4)
        masks = np.asarray(masks, dtype=bool).reshape(n_samples, len(worlds))
        scores = masks.sum(axis=1)
        best = np.argsort(-scores, kind="stable")[:max_candidates]
        if output_path is not None:
            np.savez_compressed(output_path, poses=poses, pass_mask=masks, scores=scores, selected_indices=best)
        print(f"[ensemble]: highest score {scores.max() if len(scores) else 0}/{len(worlds)}")
        return [poses[i] for i in best]

    def sample_and_validate(self, visualize=False, n_samples=1000, max_valid=10, sim_steps=250,
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


class EnsembleTwin:
    """One reusable model: 30 cm tables, fixed passives, and free active bodies."""
    def __init__(self, builder, active_name, passive_name, T_passive, T_goal, obstacles,
                 hypotheses, V, S, output_path, scales=(1.1, 1.0, 0.9)):
        from itertools import product
        self.V, self.S = V, S
        self.T_goal, self.T_passive = T_goal.copy(), T_passive.copy()
        self.specs = [dict(active_scale=a, passive_scale=p, hypothesis=h)
                      for a, p, h in product(scales, scales, range(len(hypotheses)))]
        self.n = len(self.specs)
        if not 1 <= self.n <= 31: raise ValueError("Ensemble needs 1–31 cells for collision isolation")
        lab = ET.parse(builder.lab_scene_path).getroot()
        table = lab.find(".//geom[@name='table']")
        if table is None: raise ValueError("Missing tabletop geom 'table'")
        table_pos = np.fromstring(table.get("pos", "0 0 0"), sep=" ")
        table_size = np.fromstring(table.get("size"), sep=" ")
        self.table_z = table_pos[2] + table_size[2]
        root = ET.Element("mujoco", model="validation_ensemble")
        ET.SubElement(root, "compiler", inertiafromgeom="true", fusestatic="false")
        option = copy.deepcopy(lab.find("option")) if lab.find("option") is not None else ET.Element("option")
        option.set("jacobian", "sparse")
        root.append(option)
        asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
        loaded, active_bodies, self.offsets, centers = {}, [], [], []

        def asset_elements(path):
            for elem in ET.parse(path).getroot():
                if elem.tag == "include":
                    yield from asset_elements((path.parent / elem.get("file")).resolve())
                else:
                    elem = copy.deepcopy(elem)
                    if elem.get("file"):
                        elem.set("file", str((path.parent / elem.get("file")).resolve()))
                    yield elem

        def add_body(name, T, role, cell, scale, physics):
            catalog = dict(builder.object_catalog)
            catalog[name] = dict(catalog[name], physics=physics)
            body = SceneBuilder(catalog, builder.lab_scene_path).make_body(name, T, role)
            key = (name, scale)
            if key not in loaded:
                mjcf = catalog[name]["mjcf"]
                elems = list(asset_elements(builder._resolve(mjcf["asset_xml"]))) if mjcf["type"] == "mesh" else []
                prefix = f"asset{len(loaded)}_"
                mapping = {e.get("name"): prefix + e.get("name") for e in elems if e.get("name")}
                for elem in elems:
                    for attr in ("name", "mesh", "material", "texture", "hfield"):
                        if elem.get(attr) in mapping: elem.set(attr, mapping[elem.get(attr)])
                    if elem.tag == "mesh":
                        base = np.fromstring(elem.get("scale", "1 1 1"), sep=" ")
                        elem.set("scale", " ".join(map(str, base * scale)))
                    asset.append(elem)
                loaded[key] = mapping
            for elem in body.iter():
                if elem.get("name"): elem.set("name", f"cell{cell}_" + elem.get("name"))
                for attr in ("mesh", "material", "texture", "hfield"):
                    if elem.get(attr) in loaded[key]: elem.set(attr, loaded[key][elem.get(attr)])
            for geom in body.iter("geom"):
                geom.set("contype", str(1 << cell)); geom.set("conaffinity", str(1 << cell))
            world.append(body)
            return body

        cols = int(np.ceil(np.sqrt(self.n)))
        for i, spec in enumerate(self.specs):
            center = np.array([0.55 * (i % cols), 0.55 * (i // cols)])
            offset = np.r_[center - T_passive[:2, 3], 0.0]
            centers.append(center); self.offsets.append(offset)
            attrs = {k: v for k, v in table.attrib.items() if k in (
                "friction", "condim", "solref", "solimp", "margin", "gap", "priority", "solmix")}
            ET.SubElement(world, "geom", name=f"cell{i}_table", type="box",
                          pos=f"{center[0]} {center[1]} {table_pos[2]}",
                          size=f"0.25 0.25 {table_size[2]}", contype=str(1 << i), conaffinity=str(1 << i), **attrs)
            for name, T, role, scale in ((active_name, T_goal, "active", spec["active_scale"]),
                                         (passive_name, T_passive, "passive", spec["passive_scale"])):
                pose = T.copy(); pose[:3, 3] += offset
                body = add_body(name, pose, role, i, scale, hypotheses[spec["hypothesis"]][role])
                if role == "active": active_bodies.append(body)
            for name, T in obstacles:
                pose = T.copy(); pose[:3, 3] += offset
                add_body(name, pose, "obstacle", i, 1.0, builder.object_catalog[name]["physics"])

        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(root, space="  "); ET.ElementTree(root).write(path)
        probe = mujoco.MjModel.from_xml_path(str(path))
        for body, spec in zip(active_bodies, self.specs):
            volume = float(probe.body_mass[probe.body(body.get("name")).id])
            mass = float(hypotheses[spec["hypothesis"]]["active"]["mass"])
            if volume <= 0 or mass <= 0: raise ValueError("Invalid ensemble volume/mass")
            for geom in body.iter("geom"): geom.set("density", str(mass / volume))
        del probe
        ET.ElementTree(root).write(path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.offsets, self.centers = np.asarray(self.offsets), np.asarray(centers)
        self.active = np.array([self.model.body(f"cell{i}_active_{active_name}").id for i in range(self.n)])
        self.passive = np.array([self.model.body(f"cell{i}_passive_{passive_name}").id for i in range(self.n)])
        self.qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.active]]
        self.geom_cell = np.full(self.model.ngeom, -1, dtype=int)
        self.geom_role = np.full(self.model.ngeom, -1, dtype=int)
        for i in range(self.n):
            self.geom_cell[self.model.geom(f"cell{i}_table").id] = i
            self.geom_role[self.model.geom(f"cell{i}_table").id] = 2
            for role, names, code in (("active", [active_name], 0), ("passive", [passive_name], 1),
                                       ("obstacle", [n for n, _ in obstacles], 3)):
                for name in names:
                    bid = self.model.body(f"cell{i}_{role}_{name}").id
                    descendants = {bid}
                    for b in range(bid + 1, self.model.nbody):
                        if self.model.body_parentid[b] in descendants: descendants.add(b)
                    selected = np.isin(self.model.geom_bodyid, list(descendants))
                    self.geom_cell[selected], self.geom_role[selected] = i, code

    def contacts(self):
        penetration = np.zeros(self.n)
        touching = np.zeros((self.n, 4), dtype=bool)
        for c in self.data.contact:
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if self.geom_role[a] != 0: continue
                cell = self.geom_cell[a]
                penetration[cell] = max(penetration[cell], -float(c.dist))
                other = self.geom_role[b] if self.geom_cell[b] == cell else 3
                if other >= 0: touching[cell, other] = True
        return penetration, touching

    def validate(self, pose, sim_steps=250):
        m, d, V = self.model, self.data, self.V
        mujoco.mj_resetData(m, d)
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        d.qpos[self.qadr[:, None] + np.arange(7)] = np.c_[pose[:3, 3] + self.offsets, np.tile(q, (self.n, 1))]
        mujoco.mj_forward(m, d)
        geometric = V.validation_mode == "geometric_only"
        valid = self.contacts()[0] <= (V.max_final_penetration if geometric else V.max_initial_penetration)
        if not geometric:
            try:
                mujoco.mj_step(m, d, nstep=sim_steps)
                mujoco.mj_forward(m, d)
            except mujoco.FatalError:
                return np.zeros(self.n, dtype=bool)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):
                return np.zeros(self.n, dtype=bool)
        penetration, touching = self.contacts()
        valid &= (penetration <= V.max_final_penetration) & ~touching[:, 3]
        for contact, column in (("active_passive", 1), ("active_table", 2)):
            if contact in V.required_contacts: valid &= touching[:, column]
            if contact in V.forbidden_contacts: valid &= ~touching[:, column]
        positions = d.xpos[self.active] - self.offsets
        # Falling off a small table must not evade forbidden tabletop contact.
        valid &= np.all(np.abs(d.xpos[self.active, :2] - self.centers) <= 0.25, axis=1)
        if not geometric:
            valid &= np.all(np.abs(positions - pose[:3, 3]) <= [V.x_drift_tolerance, V.y_drift_tolerance, V.z_drift_tolerance], axis=1)
            delta = pose[:3, :3].T @ d.xmat[self.active].reshape(-1, 3, 3)
            drift = np.abs(Rotation.from_matrix(delta).as_euler("xyz", degrees=True))
            valid &= np.all(drift <= [V.roll_drift_tolerance, V.pitch_drift_tolerance, V.yaw_drift_tolerance], axis=1)
        relative = d.xpos[self.active] - d.xpos[self.passive]  # world-aligned reasoning frame
        for relation in V.xyz_relations:
            value = relative[:, DigitalTwin.AXIS[relation.axis]]
            valid &= value > 0 if relation.relation == "greater_than" else value < 0
        return valid

    def rank(self, n_samples=200, max_candidates=5, sim_steps=250, output_path=None):
        if n_samples < 1 or max_candidates < 1: raise ValueError("Invalid sample/candidate count")
        geometric = self.V.validation_mode == "geometric_only"
        poses, masks, settled = [], [], []
        for i in range(n_samples):
            pose = self.T_goal.copy() if i == 0 else self.S.sample(self.T_goal, self.T_passive)
            poses.append(pose); masks.append(self.validate(pose, sim_steps))
            if not geometric:
                final = np.tile(np.eye(4), (self.n, 1, 1))
                final[:, :3, :3] = self.data.xmat[self.active].reshape(-1, 3, 3)
                final[:, :3, 3] = self.data.xpos[self.active] - self.offsets
                settled.append(final)
            if (i + 1) % 10 == 0: print(f"[ensemble]: scored {i+1}/{n_samples} proposals")
        poses, masks = np.asarray(poses), np.asarray(masks)
        scores = masks.sum(axis=1)
        ranked = np.argsort(-scores, kind="stable")
        if geometric:
            others = ranked[(ranked != 0) & (scores[ranked] > 0)][:max_candidates - 1]
            best = np.r_[0, others]
        else:
            best = ranked[scores[ranked] > 0][:max_candidates]
        candidates, worlds = [], []
        for i in best:
            world = -1 if geometric else int(np.random.choice(np.flatnonzero(masks[i])))
            candidates.append(poses[i].copy() if geometric else settled[i][world].copy())
            worlds.append(world)
        if output_path is not None:
            np.savez_compressed(output_path, poses=poses, pass_mask=masks, scores=scores, selected_indices=best,
                                selected_world_indices=np.asarray(worlds, dtype=int),
                                candidate_poses=np.asarray(candidates).reshape(-1, 4, 4))
        print(f"[ensemble]: highest score {scores.max()}/{self.n}")
        return candidates
