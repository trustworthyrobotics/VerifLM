import numpy as np
import cv2
from dataclasses import dataclass
from typing import Literal
from digital_twin import Scene
from scipy.spatial.transform import Rotation

ANYGRASP_HOVER_DISTANCE = 0.10
EEF_FINGERTIP_REACH = 0.1135

ANYGRASP_EEF_ROTATION = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=float)

@dataclass(frozen=True)
class Grasp:
    T_object_eef: np.ndarray
    T_object_pregrasp_eef: np.ndarray | None = None

@dataclass(frozen=True)
class FeasibleGrasp:
    grasp: Grasp
    q_source: np.ndarray
    q_goal: np.ndarray
    q_pregrasp: np.ndarray

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
    def __init__(self, object_catalog, K, anygrasp_checkpoint):
        self.object_catalog = object_catalog
        self.K = np.asarray(K, dtype=float)
        self.anygrasp_checkpoint = anygrasp_checkpoint

    def plan(self, source_pose, rgb_path=None, depth_path=None, mask_path=None,
             use_anygrasp=False, mask_margin_px=10, max_candidates=50):
        # Generate AnyGrasp candidates near the active-object mask or use hardcoded grasps
        if not use_anygrasp:
            return self.load_hardcoded()

        if rgb_path is None or depth_path is None or mask_path is None:
            raise ValueError("rgb_path, depth_path, and mask_path are required for AnyGrasp")

        rgb = cv2.imread(str(rgb_path))
        depth = np.load(depth_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None or mask is None:
            raise ValueError("Could not load RGB image or mask")
        if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
            raise ValueError("RGB, depth, and mask must have the same resolution")

        points, colors = self._point_cloud(rgb, depth)
        mask = self._dilate_mask(mask, mask_margin_px)
        raw = self._run_anygrasp(points, colors)

        if len(raw) == 0:
            return []

        raw = self._filter_by_mask(raw, mask)
        raw = raw[np.argsort(raw[:, 0])[::-1]]

        if max_candidates is not None:
            raw = raw[:max_candidates]

        return [self._to_grasp(row, source_pose) for row in raw]

    def _point_cloud(self, rgb, depth, min_depth=0.1, max_depth=2.0):
        # Convert RGB-D into camera-frame point cloud
        valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        v, u = np.indices(depth.shape)
        z = depth[valid]

        x = (u[valid] - self.K[0, 2]) * z / self.K[0, 0]
        y = (v[valid] - self.K[1, 2]) * z / self.K[1, 1]

        points = np.column_stack((x, y, z)).astype(np.float32)
        colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[valid].astype(np.float32) / 255.0
        return points, colors

    def _dilate_mask(self, mask, margin_px):
        # Expand mask so nearby grasp centers are accepted
        mask = mask > 0
        if margin_px <= 0:
            return mask

        size = 2 * margin_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    def _filter_by_mask(self, grasps, mask):
        # Keep grasps whose centers project inside the expanded mask
        x, y, z = grasps[:, 13], grasps[:, 14], grasps[:, 15]
        valid = z > 0

        u = np.zeros(len(grasps), dtype=int)
        v = np.zeros(len(grasps), dtype=int)
        u[valid] = np.rint(self.K[0, 0] * x[valid] / z[valid] + self.K[0, 2]).astype(int)
        v[valid] = np.rint(self.K[1, 1] * y[valid] / z[valid] + self.K[1, 2]).astype(int)

        inside = valid & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        keep = np.zeros(len(grasps), dtype=bool)
        keep[inside] = mask[v[inside], u[inside]]
        return grasps[keep]

    def _run_anygrasp(self, points, colors):
        # Run AnyGrasp and return its NMS-filtered grasp array
        from gsnet import AnyGrasp

        cfg = SimpleNamespace(
            checkpoint_path=str(self.anygrasp_checkpoint),
            max_gripper_width=0.05,
            gripper_height=0.03,
            top_down_grasp=False,
            debug=False,
        )

        detector = AnyGrasp(cfg)
        detector.load_net()

        mins, maxs = points.min(axis=0), points.max(axis=0)
        limits = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

        grasps, _ = detector.get_grasp(
            points, colors, lims=limits,
            apply_object_mask=True, dense_grasp=False, collision_detection=True,
        )

        if len(grasps) == 0:
            return np.empty((0, 17))

        return np.asarray(grasps.nms().sort_by_score().grasp_group_array, dtype=float)

    def _to_grasp(self, row, T_cam_object):
        # Convert AnyGrasp camera-frame grasp into object-relative EEF grasp
        T_cam_grasp = np.eye(4)
        T_cam_grasp[:3, :3] = row[4:13].reshape(3, 3)
        T_cam_grasp[:3, 3] = row[13:16]

        T_grasp_eef = np.eye(4)
        T_grasp_eef[:3, :3] = ANYGRASP_EEF_ROTATION
        T_grasp_eef[:3, 3] = [row[3] - EEF_FINGERTIP_REACH, 0, 0]

        T_object_eef = np.linalg.inv(T_cam_object) @ T_cam_grasp @ T_grasp_eef

        T_object_pregrasp_eef = T_object_eef.copy()
        T_object_pregrasp_eef[:3, 3] -= ANYGRASP_HOVER_DISTANCE * T_object_eef[:3, 2]

        return Grasp(T_object_eef, T_object_pregrasp_eef)    

    def load_hardcoded(self, object_catalog):
        grasps = []
        grasp_configs = object_catalog["grasps"]
        if grasp_configs == None or len(grasp_configs) == 0:
            return grasps

        for grasp_config in grasp_configs:
            offset_xyz = grasp_config["offset_xyz"]
            offset_rpy_deg = grasp_config["offset_rpy_deg"]
            hover_offset_xyz = grasp_config["hover_offset_xyz"]

            T_object_eef = np.eye(4)
            T_object_eef[:3, :3] = Rotation.from_euler("xyz", offset_rpy_deg, degrees=True).as_matrix()
            T_object_eef[:3, 3] = np.asarray(offset_xyz, dtype=float)

            T_object_pregrasp_eef = T_object_eef.copy()
            T_object_pregrasp_eef[:3, 3] += np.array(hover_offset_xyz)

            z_rotation_samples = grasp_config["z_rotation_samples"]
            if not grasp_config["z_rotation_samples"] or grasp_config["z_rotation_samples"] == 1:
                continue

            for angle in np.linspace(0.0, 360.0, z_rotation_samples, endpoint=False):
                T_object_rotation = np.eye(4)
                T_object_rotation[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()

                T_object_eef = T_object_rotation @ T_object_eef
                T_object_pregrasp_eef = T_object_rotation @ T_object_pregrasp_eef
                grasps.append(Grasp(T_object_eef=T_object_eef, T_object_pregrasp_eef=T_object_pregrasp_eef))

        return grasps

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
        self.scene = scene
        pass

    def plan(self, q_start, q_goal, candidate_grasps):
        # For every candidate grasp that passed (1) IK feasible source and goal (2) collision free at source and goal
        ## get OMPL trajectory
        ## interpolate()
        ## time-parametrize
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