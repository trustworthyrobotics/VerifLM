from dataclasses import dataclass
from typing import Literal
from digital_twin import Scene
import numpy as np

@dataclass(frozen=True)
class Grasp:
    T_object_eef: np.ndarray
    T_object_pregrasp_eef: np.ndarray | None = None

@dataclass(frozen=True)
class TrajectoryStage:
    type: Literal["move", "grasp", "release"]
    q_path: list[np.ndarray] | None = None

def solve_ik(scene: Scene, T_world_eef):
    pass

def collision_free(scene: Scene, T_eef_active):
    pass

class GraspPlanner: 
    def __init__(self):
        pass

    def plan(self, source_pose, goal_pose, rgbd, use_anygrasp=False):
        pass

    def load_hardcoded(self, offset_xyz, offset_rpy_deg, pregrasp_offset_xyz, z_samples=1):
        pass

class MotionPlanner:
    def __init__(self, scene: Scene):
        pass

    def plan(self, q_goal, q_start, T_eef_active, T_world_active):
        pass

class TrajectoryPlanner: 
    def __init__(self, scene: Scene):
        pass

    def plan(self, q_goal, q_start, T_eef_active, T_world_active):
        pass
    
    def replay_trajectory(self, trajectory: list[TrajectoryStage]):
        pass