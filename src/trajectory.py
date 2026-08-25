import numpy as np
import cv2
import mujoco
import mujoco.viewer
import json
import time
import signal, os, sys
from dataclasses import dataclass
from typing import Literal
from src.digital_twin import Scene
from scipy.spatial.transform import Rotation
from ompl import base as ob
from ompl import geometric as og
from types import SimpleNamespace
from pathlib import Path

ANYGRASP_HOVER_DISTANCE = 0.10
EEF_FINGERTIP_REACH = 0.1135
ANYGRASP_EEF_ROTATION = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=float)

ARM_PREFIX = "arm1_"
ARM_JOINT_NAMES = tuple(f"{ARM_PREFIX}joint{i}" for i in range(1, 8))
ROBOT_BODY_NAMES = tuple([ARM_PREFIX + name for name in ("base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "adapter_box", "box_gripper", "finger_1", "finger_2")])
FINGER_BODY_NAMES = (ARM_PREFIX+"finger_1", ARM_PREFIX+"finger_2")
FINGER_JOINT_NAMES = (ARM_PREFIX+"finger_1_slide", ARM_PREFIX+"finger_2_slide")
EEF_BODY_NAME = ARM_PREFIX + "box_gripper"
TABLE_GEOM_NAMES = ("table", "table_base")
ACTIVE_TABLE_TOLERANCE = 0.01
ACTIVE_PASSIVE_TOLERANCE = 0.0005

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

class CollisionChecker:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.model = scene.model
        self.data = mujoco.MjData(self.model)

        # Arm joint addresses
        self.arm_qadrs = np.array([self.model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES])

        # Active object freejoint address
        active_joint_id = int(self.model.body_jntadr[scene.active_body_id])
        self.active_qadr = int(self.model.jnt_qposadr[active_joint_id])

        # Robot geoms
        robot_body_ids = {self.model.body(name).id for name in ROBOT_BODY_NAMES}
        self.robot_geom_ids = {
            gid for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) in robot_body_ids
        }

        # Finger geoms
        finger_body_ids = {self.model.body(name).id for name in FINGER_BODY_NAMES}
        self.finger_geom_ids = {gid for gid in range(self.model.ngeom) if int(self.model.geom_bodyid[gid]) in finger_body_ids}

        # Active/passive object geoms
        self.active_geom_ids = {gid for gid in range(self.model.ngeom) if int(self.model.geom_bodyid[gid]) == scene.active_body_id}
        self.passive_geom_ids = {gid for gid in range(self.model.ngeom) if int(self.model.geom_bodyid[gid]) == scene.passive_body_id}

        self.table_geom_ids = {self.model.geom(name).id for name in TABLE_GEOM_NAMES}
        self.eef_body_id = self.model.body(EEF_BODY_NAME).id

    def check(self, q, T_eef_active=None, allow_active_fingers=False):
        # Set arm to q and attach active object to EEF if provided
        self.data.qpos[:] = self.scene.data.qpos
        self.data.qvel[:] = 0
        self.data.qpos[self.arm_qadrs] = q
        mujoco.mj_forward(self.model, self.data)

        if T_eef_active is not None:
            eef = self.data.body(EEF_BODY_NAME)

            T_world_eef = np.eye(4)
            T_world_eef[:3, 3] = eef.xpos
            T_world_eef[:3, :3] = eef.xmat.reshape(3, 3)

            T_world_active = T_world_eef @ T_eef_active
            quat_xyzw = Rotation.from_matrix(T_world_active[:3, :3]).as_quat()

            self.data.qpos[self.active_qadr:self.active_qadr + 3] = T_world_active[:3, 3]
            self.data.qpos[self.active_qadr + 3:self.active_qadr + 7] = quat_xyzw[[3, 0, 1, 2]]

            mujoco.mj_forward(self.model, self.data)

        # Check contacts
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)

            if contact.dist >= 0:
                continue

            r1, r2 = g1 in self.robot_geom_ids, g2 in self.robot_geom_ids
            a1, a2 = g1 in self.active_geom_ids, g2 in self.active_geom_ids
            p1, p2 = g1 in self.passive_geom_ids, g2 in self.passive_geom_ids
            t1, t2 = g1 in self.table_geom_ids, g2 in self.table_geom_ids

            # Small active-table penetration is tolerated
            if (a1 and t2) or (a2 and t1):
                if contact.dist >= -ACTIVE_TABLE_TOLERANCE:
                    continue
                return False

            # Small active-passive penetration is tolerated
            if (a1 and p2) or (a2 and p1):
                if contact.dist >= -ACTIVE_PASSIVE_TOLERANCE:
                    continue
                return False

            # Robot self-collision
            if r1 and r2:
                return False

            # Active object may contact fingers only during grasp/transfer stages
            if (a1 and r2) or (a2 and r1):
                robot_geom = g2 if a1 else g1

                if allow_active_fingers and robot_geom in self.finger_geom_ids:
                    continue

                return False

            # Robot-environment collision
            if r1 or r2:
                return False

            # Attached active object hitting anything else
            if T_eef_active is not None and (a1 or a2):
                return False

        return True

class StateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, collision_checker, T_eef_active=None):
        super().__init__(si)
        self.collision_checker = collision_checker
        self.T_eef_active = T_eef_active

    def isValid(self, state):
        q = np.array([state[i] for i in range(7)])
        return self.collision_checker.check(
            q,
            self.T_eef_active,
            allow_active_fingers=self.T_eef_active is not None,
        )        

class GraspPlanner:
    def __init__(self, object_catalog, K, anygrasp_checkpoint):
        self.object_catalog = object_catalog
        self.K = np.asarray(K, dtype=float)
        self.anygrasp_checkpoint = anygrasp_checkpoint

    def plan(self, T_cam_object, rgb_path=None, depth_path=None, mask_path=None,
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

        return [self._to_grasp(row, T_cam_object) for row in raw]

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

    def load_hardcoded(self):
        grasps = []
        grasp_configs = self.object_catalog["grasps"]
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

            z_rotation_samples = grasp_config.get("z_rotation_samples", 1)
            if z_rotation_samples is None or z_rotation_samples == 1:
                grasps.append(Grasp(T_object_eef, T_object_pregrasp_eef))
                continue

            for angle in np.linspace(0.0, 360.0, z_rotation_samples, endpoint=False):
                T_object_rotation = np.eye(4)
                T_object_rotation[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
                grasps.append(Grasp(T_object_rotation @ T_object_eef, T_object_rotation @ T_object_pregrasp_eef))

        return grasps

class TrajectoryPlanner: 
    def __init__(self, scene: Scene, max_joint_vel=0.2):
        self.scene = scene
        self.model = scene.model
        self.ik_data = mujoco.MjData(self.model)
        self.max_joint_vel = max_joint_vel
        self.collision_checker = CollisionChecker(scene)
        self.arm_joint_limits = np.array([self.model.joint(name).range for name in ARM_JOINT_NAMES])

        self.arm_qadrs = np.array([self.model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES])
        self.arm_dadrs = np.array([self.model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES])
        self.eef_body_id = self.model.body(EEF_BODY_NAME).id
        self.chosen_grasp = None

        self.finger_qadrs = np.array([self.model.joint(name).qposadr[0] for name in FINGER_JOINT_NAMES])

    def plan(self, q_start, T_mujoco_object_src, T_mujoco_object_goal, grasps: list[Grasp],
         preinsertion_offset=None, output_path=None):
        print(f"T_mujoco_object_src:\n{T_mujoco_object_src}, T_mujoco_object_goal:\n{T_mujoco_object_goal}")

        offset = None if preinsertion_offset is None else np.asarray(preinsertion_offset, dtype=float)
        if offset is not None and offset.shape != (3,):
            raise ValueError("preinsertion_offset must contain exactly 3 floats")

        stages = None
        for i, grasp in enumerate(grasps):
            T_eef_object = np.linalg.inv(grasp.T_object_eef)
            T_eef_grasp = T_mujoco_object_src @ grasp.T_object_eef
            T_eef_pregrasp = T_mujoco_object_src @ grasp.T_object_pregrasp_eef
            T_eef_goal = T_mujoco_object_goal @ grasp.T_object_eef

            T_object_preinsert = None
            if offset is not None:
                T_object_preinsert = T_mujoco_object_goal.copy()
                T_object_preinsert[:3, 3] += offset       # world-frame displacement
                T_eef_preinsert = T_object_preinsert @ grasp.T_object_eef

            q_pregrasp = self.solve_ik(T_eef_pregrasp)
            if q_pregrasp is None:
                print(f"[trajectory]: {i}/{len(grasps)} failed IK for pregrasp")
                continue

            q_grasp = self.solve_ik(T_eef_grasp, q_pregrasp)
            if q_grasp is None:
                print(f"[trajectory]: {i}/{len(grasps)} failed IK for grasp")
                continue

            q_preinsert = self.solve_ik(T_eef_preinsert, q_grasp) if offset is not None else None
            if offset is not None and q_preinsert is None:
                print(f"[trajectory]: {i}/{len(grasps)} failed IK for preinsertion")
                continue

            q_goal = self.solve_ik(T_eef_goal, q_preinsert if q_preinsert is not None else q_grasp)
            if q_goal is None:
                print(f"[trajectory]: {i}/{len(grasps)} failed IK for goal")
                continue

            # Home -> pregrasp -> grasp
            p0 = self.plan_trajectory(q_start, q_pregrasp)
            if not p0:
                print(f"[trajectory]: {i}/{len(grasps)} failed start -> pregrasp")
                continue

            p1 = self.plan_grasp_approach(q_pregrasp, q_grasp)
            if not p1:
                print(f"[trajectory]: {i}/{len(grasps)} failed pregrasp -> grasp")
                continue

            p2 = list(reversed(p1))  # grasp -> source pregrasp

            # Transfer attached object.
            q_transfer_goal = q_preinsert if q_preinsert is not None else q_goal
            p3 = self.plan_trajectory(q_pregrasp, q_transfer_goal, T_eef_object)
            if not p3:
                name = "preinsertion" if offset is not None else "goal"
                print(f"[trajectory]: {i}/{len(grasps)} failed pregrasp -> {name}")
                continue

            # Optional constrained final insertion.
            p4 = None
            if offset is not None:
                p4 = self.plan_insertion_approach(q_preinsert, q_goal, T_eef_object)
                if not p4:
                    print(f"[trajectory]: {i}/{len(grasps)} failed preinsertion -> goal")
                    continue

            paths = [p0, p1, p2, p3]
            stages = [
                TrajectoryStage("move", q_path=p0, time_from_start=self.time_parametrize(p0)),
                TrajectoryStage("move", q_path=p1, time_from_start=self.time_parametrize(p1)),
                TrajectoryStage("grasp"),
                TrajectoryStage("move", q_path=p2, time_from_start=self.time_parametrize(p2)),
                TrajectoryStage("move", q_path=p3, time_from_start=self.time_parametrize(p3)),
            ]
            if p4 is not None:
                paths.append(p4)
                stages.append(TrajectoryStage("move", q_path=p4, time_from_start=self.time_parametrize(p4)))
            stages.append(TrajectoryStage("release"))

            self.chosen_grasp = grasp
            break

        if output_path is not None and stages is not None:
            result = {
                "T_object_eef": self.chosen_grasp.T_object_eef.tolist(),
                "T_object_pregrasp_eef": self.chosen_grasp.T_object_pregrasp_eef.tolist(),
                "preinsertion_offset": offset.tolist() if offset is not None else None,
                "trajectory": [{
                    "type": s.type,
                    "q_path": [q.tolist() for q in s.q_path] if s.q_path is not None else None,
                    "time_from_start": s.time_from_start,
                } for s in stages],
            }
            Path(output_path).write_text(json.dumps(result, indent=2))

        return stages


    def plan_insertion_approach(self, q_preinsertion, q_goal, T_eef_active, steps=30):
        path = [(1-a)*q_preinsertion + a*q_goal for a in np.linspace(0, 1, steps)]
        for q in path:
            if not self.collision_checker.check(q, T_eef_active, allow_active_fingers=True):
                return None
        return path

    def plan_trajectory(self, q_start, q_goal, T_eef_active=None, solve_time=5.0):
        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for i, (low, high) in enumerate(self.arm_joint_limits):
            bounds.setLow(i, low)
            bounds.setHigh(i, high)
        space.setBounds(bounds)

        setup = og.SimpleSetup(space)
        setup.setStateValidityChecker(StateValidityChecker(setup.getSpaceInformation(), self.collision_checker, T_eef_active))
        setup.getSpaceInformation().setStateValidityCheckingResolution(0.005)

        start = space.allocState()
        goal = space.allocState()

        for i in range(7):
            start[i] = q_start[i]
            goal[i] = q_goal[i]

        setup.setStartAndGoalStates(start, goal)

        setup.setStartAndGoalStates(start, goal)
        setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))

        if not setup.solve(solve_time):
            return None

        setup.simplifySolution()
        path = setup.getSolutionPath()
        path.interpolate()
        q_list = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q_list.append(np.array([state[j] for j in range(7)], dtype=float))
        return q_list

    def plan_grasp_approach(self, q_pregrasp, q_grasp, steps=15):
        path = [(1-a)*q_pregrasp+a*q_grasp for a in np.linspace(0, 1, steps)]
        return path

    def time_parametrize(self, q_path, min_segment_dt=0.001):
        times = [0.0]
        elapsed = 0.0
        for q0, q1 in zip(q_path[:-1], q_path[1:]):
            delta_q = np.abs(q1 - q0)
            segment_dt = np.max(delta_q) / self.max_joint_vel
            segment_dt = max(segment_dt, min_segment_dt)
            elapsed += segment_dt
            times.append(elapsed)
        return times
    
    def solve_ik(self, T_mujoco_eef, q_init=None, max_iters=500, pos_tol=1e-3, rot_tol=1e-2, damping=1e-3, step_size=0.5):
        self.ik_data.qpos[:] = self.scene.data.qpos
        q = self.ik_data.qpos[self.arm_qadrs].copy() if q_init is None else np.asarray(q_init).copy()

        for _ in range(max_iters):
            q = np.clip(q, self.arm_joint_limits[:,0], self.arm_joint_limits[:,1])
            self.ik_data.qpos[self.arm_qadrs] = q
            mujoco.mj_forward(self.model, self.ik_data)

            eef = self.ik_data.body(EEF_BODY_NAME)
            pos_err = T_mujoco_eef[:3,3] - eef.xpos
            rot_err = Rotation.from_matrix(T_mujoco_eef[:3,:3] @ eef.xmat.reshape(3,3).T).as_rotvec()
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                return q.copy()

            jac_pos, jac_rot = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.ik_data, jac_pos, jac_rot, self.eef_body_id)
            J = np.vstack((jac_pos[:, self.arm_dadrs], jac_rot[:, self.arm_dadrs]))
            error = np.hstack((pos_err, rot_err))
            q += step_size * J.T @ np.linalg.solve(J @ J.T + damping*np.eye(6), error)

        return None

    def replay_trajectory(self, trajectory: list[TrajectoryStage]):
        data = mujoco.MjData(self.model)
        active_qadr = self.collision_checker.active_qadr
        T_eef_object = np.linalg.inv(self.chosen_grasp.T_object_eef)

        stop = {"requested": False}
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(
            signal.SIGINT,
            lambda *_: stop.__setitem__("requested", True),
        )

        def move_gripper(target, viewer, duration=.5):
            start = data.qpos[self.finger_qadrs].copy()
            for a in np.linspace(0, 1, 30):
                if stop["requested"] or not viewer.is_running():
                    return
                data.qpos[self.finger_qadrs] = (1-a)*start + a*target
                mujoco.mj_forward(self.model, data)
                viewer.sync()
                time.sleep(duration/30)

        try:
            with mujoco.viewer.launch_passive(self.model, data) as viewer:
                while viewer.is_running() and not stop["requested"]:
                    data.qpos[:] = self.scene.data.qpos
                    data.qvel[:] = 0
                    data.mocap_pos[:] = self.scene.data.mocap_pos
                    data.mocap_quat[:] = self.scene.data.mocap_quat
                    mujoco.mj_forward(self.model, data)
                    attached = False

                    for stage in trajectory:
                        if stop["requested"] or not viewer.is_running():
                            break

                        if stage.type == "grasp":
                            move_gripper(0.025, viewer)
                            attached = True
                            continue

                        if stage.type == "release":
                            attached = False
                            move_gripper(0.0, viewer)
                            continue

                        t0 = time.monotonic()
                        for q, t in zip(stage.q_path, stage.time_from_start):
                            if stop["requested"] or not viewer.is_running():
                                break

                            t /= 3.0
                            while (
                                not stop["requested"]
                                and viewer.is_running()
                                and time.monotonic() - t0 < t
                            ):
                                viewer.sync()
                                time.sleep(0.001)

                            if stop["requested"] or not viewer.is_running():
                                break

                            data.qpos[self.arm_qadrs] = q
                            mujoco.mj_forward(self.model, data)

                            if attached:
                                eef = data.body(EEF_BODY_NAME)
                                T_mujoco_eef = np.eye(4)
                                T_mujoco_eef[:3, 3] = eef.xpos
                                T_mujoco_eef[:3, :3] = eef.xmat.reshape(3, 3)

                                T_mujoco_object = T_mujoco_eef @ T_eef_object
                                quat = Rotation.from_matrix(
                                    T_mujoco_object[:3, :3]
                                ).as_quat()

                                data.qpos[active_qadr:active_qadr+3] = T_mujoco_object[:3, 3]
                                data.qpos[active_qadr+3:active_qadr+7] = quat[[3, 0, 1, 2]]
                                mujoco.mj_forward(self.model, data)

                            viewer.sync()

        finally:
            signal.signal(signal.SIGINT, old_handler)

        print("[trajectory]: replay exited cleanly")
        sys.stdout.flush()
        os._exit(0)