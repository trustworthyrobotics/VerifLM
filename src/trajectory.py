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
    time_from_start: list[float] | None = None

def solve_ik(scene: Scene, T_world_eef):
    pass

class CollisionChecker:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.arm_qadrs = ...
        self.robot_geom_ids = ...
        self.active_geom_ids = ...
        self.passive_geom_ids = ...
        self.finger_geom_ids = ...

    def check(self, q, T_eef_active=None):
        # Set arm to q and attach active object to EEF if provided
        # Reject self-collision, robot-environment collision, and active-environment collision
        # Allow active-object contact with the gripper fingers
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
        # Initialize collision checker and cache arm joint limits
        self.scene = scene
        self.collision_checker = CollisionChecker(scene)
        self.joint_limits = ...

    def plan(self, q_start, q_goal, T_eef_active=None):
        # Run OMPL RRT with CollisionChecker.check as the state validity checker
        # Simplify and interpolate the collision-free path
        pass


class TrajectoryPlanner: 
    def __init__(self, scene: Scene, max_joint_vel=0.2):
        pass

    def plan(self, q_start, q_goal, T_eef_active, T_world_active):
        # get OMPL trajectory
        # interpolate()
        # time-parametrize
        pass

    def time_parametrize(self, q_path, min_segment_dt=0.001):
        times = [0.0]
        elapsed = [0.0]
        for q0, q1 in zip(q_path[:-1], q_path[1:]):
            delta_q = np.abs(q1 - q0)
            segment_dt = np.max(delta_q) / self.max_joint_vel
            segment_dt = max(segment_dt, min_segment_dt)
            elapsed += segment_dt
            times.append(elapsed)
        return times

    def replay_trajectory(self, trajectory: list[TrajectoryStage]):
        pass