import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Literal
from scipy.spatial.transform import Rotation

POSITION_TOLERANCE_MAP = {"strict": 0.10,"moderate": 0.25, "permissive": 0.50} # multiply by the active object's longest dimension
ORIENTATION_TOLERANCE_MAP = {"strict": 7.5, "moderate": 30.0,"permissive": 60.0} 
POSITION_CONSTRAINT_MAP = {"correct": 0.2, "normal_search": 1.0, "major_revision": 2.0} # multiply by the active object's longest dimension
ORIENTATION_CONSTRAINT_MAP = {"correct": 12.0, "normal_search": 60.0, "major_revision": 120.0} 

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
    def sample_dt(r, bias):
        if bias == "positive":
            return np.random.uniform(0, r)
        if bias == "negative":
            return np.random.uniform(-r, 0)
        return np.random.uniform(-r, r)
    
    def sample(self, T_goal): # assumes T_goal in the world frame
        dx = self.sample_dt(self.x, self.x_bias)
        dy = self.sample_dt(self.y, self.y_bias)
        dz = self.sample_dt(self.z, self.z_bias)
        dr = np.random.uniform(-self.roll, self.roll)
        dp = np.random.uniform(-self.pitch, self.pitch)
        dyaw = np.random.uniform(-self.yaw, self.yaw)

        p_delta = [dx, dy, dz]
        R_delta = Rotation.from_euler("xyz", [dr, dp, dyaw], degrees=True).as_matrix()

        T_goal_new = T_goal.copy()
        T_goal_new[0:3, 3] += p_delta # p_delta applied in world frame
        T_goal_new[0:3, 0:3] = T_goal[0:3, 0:3] @ R_delta # R_delta applied in object's own frame

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
    max_initial_penetration: float = 0.1
    max_final_penetration: float = 0.002
    reasoning: str

@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    active_body_id: int
    passive_body_id: int
    goal_qadr: int
    table_geom_ids: frozenset[int]
    goal_pose: np.ndarray

class SceneBuilder:
    def __init__(self, lab_scene_path="models/lab.xml"):
        pass

    def build(self):
        pass

class DigitalTwin:
    def __init__(self, V: ValidationProgram, S: SamplingConstraints):
        pass

    def sample_and_validate(self, visualize=False):
        pass

    def validate(self, T_sample, program):
        return False