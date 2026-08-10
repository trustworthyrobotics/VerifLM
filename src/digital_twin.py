import numpy as np
import mujoco
from dataclasses import dataclass

@dataclass(frozen=True)
class SamplingSpace:
    xyz_range: np.ndarray
    rpy_range_deg: np.ndarray
    xyz_bias: tuple[str | None, str | None, str | None] = (None, None, None)
    def sample(self, T_goal):
        return T_goal

@dataclass(frozen=True)
class ValidationProgram:
    mode: str = "settling"
    required_contacts: tuple[str, ...] = ()
    forbidden_contacts: tuple[str, ...] = ()
    max_initial_penetration: float = 0.1
    max_final_penetration: float = 0.002
    max_position_drift: np.ndarray | None = None
    max_orientation_drift_deg: float | None = None
    xyz_relations: tuple = ()

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
    def __init__(self):
        pass

    def sample_and_validate(self, visualize=False):
        pass

    def validate(self, T_sample, program):
        return False