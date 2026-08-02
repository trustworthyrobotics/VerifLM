from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import mujoco
import numpy as np

from pipeline.config import (
    ARM_JOINT_NAMES,
    EEF_BODY_NAME,
    GRIPPER_ACTUATOR_SUFFIX,
    LIFT_HEIGHT_M,
    OMPL_INTERPOLATE_COUNT,
    OMPL_SOLVE_TIME,
    OPEN_CTRL_VALUE,
    STAGE_INTERPOLATE_COUNT,
    PipelineConfig,
)
from pipeline.core import (
    ObjectPose, mat_to_quat_wxyz, quat_wxyz_to_mat,
)
from pipeline.grasp import (
    AnyGraspConfig,
    GraspCandidate,
    GraspObservation,
    GoalFilteredGrasp,
    grasp_candidate_batches,
    sample_equivalent_poses_about_z,
)


# Low-level kinematics, collision, and OMPL primitives
def _result(status, reason=None, **values):
    result = {"status": status, **values}
    if reason is not None:
        result["reason"] = reason
    return result


def get_id(m, objtype, name):
    idx = mujoco.mj_name2id(m, objtype, name)
    if idx < 0:
        raise ValueError(f"MuJoCo name not found: {name}")
    return idx


def find_name_id_by_suffix(m, objtype, suffix: str):
    counts = {
        mujoco.mjtObj.mjOBJ_BODY: m.nbody,
        mujoco.mjtObj.mjOBJ_JOINT: m.njnt,
        mujoco.mjtObj.mjOBJ_GEOM: m.ngeom,
        mujoco.mjtObj.mjOBJ_ACTUATOR: m.nu,
    }
    for obj_id in range(counts.get(objtype, 0)):
        name = mujoco.mj_id2name(m, objtype, obj_id) or ""
        if name.endswith(suffix):
            return obj_id
    return None


def get_gripper_control_info(m, actuator_suffix: str = "gripper"):
    actuator_id = find_name_id_by_suffix(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_suffix)
    if actuator_id is None:
        return {
            "actuator_id": None,
            "actuator_name": None,
            "open_ctrl": 0.0,
            "close_ctrl": 0.0,
        }

    ctrl_low, ctrl_high = m.actuator_ctrlrange[actuator_id]
    return {
        "actuator_id": int(actuator_id),
        "actuator_name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
        "open_ctrl": float(ctrl_low),
        "close_ctrl": float(ctrl_high),
    }


def set_gripper_command(m, d, actuator_id, command: float) -> None:
    if actuator_id is not None:
        d.ctrl[int(actuator_id)] = float(command)



def set_gripper_qpos_by_command(
    m,
    d,
    command: float,
    left_joint_suffix: str = "egk_prismatic_1",
    right_joint_suffix: str = "egk_prismatic_2",
) -> None:
    left_jid = find_name_id_by_suffix(m, mujoco.mjtObj.mjOBJ_JOINT, left_joint_suffix)
    right_jid = find_name_id_by_suffix(m, mujoco.mjtObj.mjOBJ_JOINT, right_joint_suffix)

    if left_jid is not None:
        d.qpos[m.jnt_qposadr[left_jid]] = float(command)
    if right_jid is not None:
        d.qpos[m.jnt_qposadr[right_jid]] = -float(command)
    if left_jid is None and right_jid is None:
        for suffix in ("finger_1_slide", "finger_2_slide"):
            jid = find_name_id_by_suffix(m, mujoco.mjtObj.mjOBJ_JOINT, suffix)
            if jid is None:
                continue
            lo, hi = m.jnt_range[jid]
            d.qpos[m.jnt_qposadr[jid]] = np.clip(0.5 * float(command), lo, hi)


def get_body_transform(m, d, body_name):
    bid = get_id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    T = np.eye(4)
    T[:3, 3] = d.xpos[bid]
    T[:3, :3] = d.xmat[bid].reshape(3, 3)
    return T


def set_freejoint_pose_by_name(m, d, joint_name, T_world_body):
    jid = get_id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr:qadr + 3] = T_world_body[:3, 3]
    d.qpos[qadr + 3:qadr + 7] = mat_to_quat_wxyz(T_world_body[:3, :3])


def get_joint_info(m, joint_names):
    qpos_ids, dof_ids, limits = [], [], []
    for name in joint_names:
        jid = get_id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_ids.append(m.jnt_qposadr[jid])
        dof_ids.append(m.jnt_dofadr[jid])
        limits.append(tuple(m.jnt_range[jid]) if m.jnt_limited[jid] else (-np.pi, np.pi))
    return np.array(qpos_ids), np.array(dof_ids), np.asarray(limits, dtype=float)


def set_arm_q(d, qpos_ids, q):
    for i, qid in enumerate(qpos_ids):
        d.qpos[qid] = q[i]


def get_arm_q(d, qpos_ids):
    return np.array([d.qpos[i] for i in qpos_ids], dtype=float)


def interpolate_joint_path(*configs, steps_per_segment=45):
    configs = [np.asarray(q, dtype=float) for q in configs if q is not None]
    if len(configs) < 2:
        return [configs[0].copy()] if configs else []

    return [
        (1.0 - alpha) * a + alpha * b
        for a, b in zip(configs[:-1], configs[1:])
        for alpha in np.linspace(0.0, 1.0, int(steps_per_segment), endpoint=False)
    ] + [configs[-1].copy()]


def clamp_to_limits(q, limits, margin=1e-6):
    q = np.asarray(q, dtype=float).copy()
    for i, (lo, hi) in enumerate(limits):
        q[i] = np.clip(q[i], lo + margin, hi - margin)
    return q


def rotation_error(R_current, R_target):
    R_err = R_target @ R_current.T
    return 0.5 * np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])


def solve_ik(m, d, qpos_ids, dof_ids, limits, eef_body_name, T_world_eef_target, q_init,
             max_iters=500, tol_pos=1e-3, tol_rot=3e-2, damping=1e-3, step_scale=0.5):
    q = clamp_to_limits(q_init, limits)
    eef_id = get_id(m, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
    target_pos, target_R = T_world_eef_target[:3, 3], T_world_eef_target[:3, :3]
    for _ in range(max_iters):
        set_arm_q(d, qpos_ids, q)
        mujoco.mj_forward(m, d)
        T = get_body_transform(m, d, eef_body_name)
        err = np.r_[target_pos - T[:3, 3], rotation_error(T[:3, :3], target_R)]
        if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
            return clamp_to_limits(q, limits), True
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, eef_id)
        J = np.vstack([jacp, jacr])[:, dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        q = clamp_to_limits(q + step_scale * dq, limits)
    return clamp_to_limits(q, limits), False


def _contact_entity_names(m, geom_id):
    body_id = int(m.geom_bodyid[geom_id])
    return (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "",
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or "",
    )


def _contact_entity_matches(names, substring):
    return any(substring in name for name in names)


def _is_allowed_contact(m, geom1, geom2, allowed_pairs):
    names1 = _contact_entity_names(m, geom1)
    names2 = _contact_entity_names(m, geom2)
    return any(
        (_contact_entity_matches(names1, a) and _contact_entity_matches(names2, b))
        or (_contact_entity_matches(names2, a) and _contact_entity_matches(names1, b))
        for a, b in allowed_pairs
    )


def _is_ignored_contact(m, geom1, geom2, ignored_body_substrings):
    body_names = (_contact_entity_names(m, geom1)[1], _contact_entity_names(m, geom2)[1])
    return any(
        substring in body_name
        for substring in ignored_body_substrings
        for body_name in body_names
    )


PASSIVE_GRIPPER_ALLOWED_PENETRATION = 0.001
PASSIVE_GRIPPER_BODIES = {
    "arm1_finger_1",
    "arm1_finger_2",
    "arm1_box_gripper",
}
PASSIVE_OBJECT_BODIES = {
    "green_dish",
    "dish_rack",
    "spoon_rack",
}


def _contact_body_names(m, geom1, geom2):
    return _contact_entity_names(m, geom1)[1], _contact_entity_names(m, geom2)[1]


def _contact_involves_body(m, geom1, geom2, target_body_names):
    contact_body_names = _contact_body_names(m, geom1, geom2)
    return any(name in contact_body_names for name in target_body_names)


def _is_passive_gripper_contact(m, geom1, geom2, active_body_names=()):
    body1_name, body2_name = _contact_body_names(m, geom1, geom2)
    active_names = set(active_body_names)
    if body1_name in active_names or body2_name in active_names:
        return False
    return (
        body1_name in PASSIVE_GRIPPER_BODIES
        and body2_name in PASSIVE_OBJECT_BODIES
    ) or (
        body2_name in PASSIVE_GRIPPER_BODIES
        and body1_name in PASSIVE_OBJECT_BODIES
    )


def _contact_penetration_threshold(
    m,
    geom1,
    geom2,
    allowed_penetration,
    penetration_body_names=(),
):
    if _is_passive_gripper_contact(
        m, geom1, geom2, active_body_names=penetration_body_names
    ):
        return PASSIVE_GRIPPER_ALLOWED_PENETRATION
    if _contact_involves_body(m, geom1, geom2, penetration_body_names):
        return float(allowed_penetration)
    return 0.0


def _collision_free(
    m,
    d,
    allowed_pairs,
    allowed_penetration=0.003,
    ignored_body_substrings=(),
    penetration_body_names=(),
):
    for i in range(d.ncon):
        c = d.contact[i]
        if _is_ignored_contact(
            m, c.geom1, c.geom2, ignored_body_substrings
        ):
            continue
        if _is_allowed_contact(m, c.geom1, c.geom2, allowed_pairs):
            continue
        threshold = _contact_penetration_threshold(
            m,
            c.geom1,
            c.geom2,
            allowed_penetration,
            penetration_body_names,
        )
        if c.dist < -threshold:
            return False
    return True


def _collision_failure_reason(
    m,
    d,
    allowed_pairs,
    allowed_penetration=0.003,
    ignored_body_substrings=(),
    penetration_body_names=(),
):
    for i in range(d.ncon):
        c = d.contact[i]
        if _is_ignored_contact(
            m, c.geom1, c.geom2, ignored_body_substrings
        ):
            continue
        geom1_name, body1_name = _contact_entity_names(m, int(c.geom1))
        geom2_name, body2_name = _contact_entity_names(m, int(c.geom2))
        if _is_allowed_contact(m, c.geom1, c.geom2, allowed_pairs):
            continue
        threshold = _contact_penetration_threshold(
            m,
            c.geom1,
            c.geom2,
            allowed_penetration,
            penetration_body_names,
        )
        if c.dist >= -threshold:
            continue

        return (
            f"collision contact[{i}] "
            f"{body1_name or '<unnamed body>'}/{geom1_name or '<unnamed geom>'} "
            f"<-> "
            f"{body2_name or '<unnamed body>'}/{geom2_name or '<unnamed geom>'}; "
            f"dist={c.dist:.6f}, threshold={threshold:.6f}, "
            f"pos={np.round(c.pos, 5)}"
        )
    return None


def collision_free(
    m,
    d,
    allowed_contact_substrings=(),
    allowed_penetration=0.003,
    ignored_body_substrings=(),
    penetration_body_names=(),
    debug=True,
):
    """Return whether the current forwarded MuJoCo state is collision-free."""
    allowed_pairs = tuple(allowed_contact_substrings)
    ignored_bodies = tuple(ignored_body_substrings)
    penetration_bodies = tuple(penetration_body_names)
    active_threshold = float(allowed_penetration)
    collision_found = False

    if debug:
        print(
            f"[collision] contacts={d.ncon}, "
            f"active_allowed_penetration={active_threshold:.6f}, "
            f"passive_gripper_allowed_penetration="
            f"{PASSIVE_GRIPPER_ALLOWED_PENETRATION:.6f}, "
            f"penetration_bodies={penetration_bodies}"
        )

    for i in range(d.ncon):
        contact = d.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_name = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            or f"<unnamed geom {geom1}>"
        )
        geom2_name = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            or f"<unnamed geom {geom2}>"
        )
        body1_id = int(m.geom_bodyid[geom1])
        body2_id = int(m.geom_bodyid[geom2])
        body1_name = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            or f"<unnamed body {body1_id}>"
        )
        body2_name = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            or f"<unnamed body {body2_id}>"
        )
        allowed = _is_allowed_contact(m, geom1, geom2, allowed_pairs)
        ignored = _is_ignored_contact(m, geom1, geom2, ignored_bodies)
        active_contact = _contact_involves_body(
            m, geom1, geom2, penetration_bodies
        )
        passive_gripper_contact = _is_passive_gripper_contact(
            m, geom1, geom2, active_body_names=penetration_bodies
        )
        threshold = _contact_penetration_threshold(
            m,
            geom1,
            geom2,
            active_threshold,
            penetration_bodies,
        )
        penetrating = contact.dist < -threshold
        rejected = penetrating and not allowed and not ignored
        collision_found = collision_found or rejected

        if debug and rejected:
            print(
                f"[collision] contact[{i}] "
                f"{body1_name}/{geom1_name} <-> {body2_name}/{geom2_name} "
                f"dist={contact.dist:.6f}, threshold={threshold:.6f}, "
                f"active_contact={active_contact}, allowed={allowed}, ignored={ignored}, "
                f"passive_gripper_contact={passive_gripper_contact}, "
                f"rejected={rejected}, pos={np.round(contact.pos, 5)}"
            )

    if debug:
        print(
            "[collision] result="
            + ("IN COLLISION" if collision_found else "COLLISION FREE")
        )
    return not collision_found


def _apply_attached_state(m, d, q, qpos_ids, eef_body_name, object_freejoint_name, T_eef_object):
    set_arm_q(d, qpos_ids, q)
    mujoco.mj_forward(m, d)
    T_world_eef = get_body_transform(m, d, eef_body_name)
    set_freejoint_pose_by_name(m, d, object_freejoint_name, T_world_eef @ T_eef_object)
    mujoco.mj_forward(m, d)


def plan_ompl_motion(
    m, d, q_start, q_goal, qpos_ids, limits, eef_body_name, object_freejoint_name,
    T_eef_object, allowed_pairs, solve_time=5.0, interpolate_count=120,
    max_z=None, xy_bounds=None,
    ignored_body_substrings=(), allowed_penetration=0.003, penetration_body_names=(),
    fixed_object_T=None,
):
    from ompl import base as ob
    from ompl import geometric as og

    allowed_pairs = tuple(allowed_pairs)
    ignored_body_substrings = tuple(ignored_body_substrings)
    penetration_body_names = tuple(penetration_body_names)
    fixed_object_T = None if fixed_object_T is None else np.asarray(fixed_object_T, dtype=float)

    ndim = len(q_start)
    space = ob.RealVectorStateSpace(ndim)
    bounds = ob.RealVectorBounds(ndim)
    for i, (lo, hi) in enumerate(limits):
        bounds.setLow(i, float(lo)); bounds.setHigh(i, float(hi))
    space.setBounds(bounds)
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()

    # Body IDs for every joint in qpos_ids 鈥?used for height constraint
    qpos_id_set = set(qpos_ids)
    arm_body_ids = list({
        int(m.jnt_bodyid[j])
        for j in range(m.njnt)
        if m.jnt_qposadr[j] in qpos_id_set
    })

    # Class-based validity checker (required by OMPL API)
    class AttachedObjectChecker(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)

        def reason(self, state) -> str | None:
            q = clamp_to_limits(np.array([state[i] for i in range(ndim)], dtype=float), limits)
            if T_eef_object is None:
                set_arm_q(d, qpos_ids, q)
                if fixed_object_T is not None:
                    set_freejoint_pose_by_name(
                        m, d, object_freejoint_name, fixed_object_T
                    )
                mujoco.mj_forward(m, d)
            else:
                _apply_attached_state(
                    m, d, q, qpos_ids, eef_body_name,
                    object_freejoint_name, T_eef_object,
                )
            if max_z is not None:
                high_bodies = [
                    (
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
                        or f"<unnamed body {bid}>",
                        float(d.xpos[bid, 2]),
                    )
                    for bid in arm_body_ids
                    if d.xpos[bid, 2] > max_z
                ]
                if high_bodies:
                    name, z = high_bodies[0]
                    return f"{name} z={z:.6f} exceeds max_z={float(max_z):.6f}"
            if xy_bounds is not None:
                x_min, x_max, y_min, y_max = xy_bounds
                for bid in arm_body_ids:
                    x = float(d.xpos[bid, 0])
                    y = float(d.xpos[bid, 1])
                    if x < x_min or x > x_max or y < y_min or y > y_max:
                        name = (
                            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
                            or f"<unnamed body {bid}>"
                        )
                        return (
                            f"{name} xy=({x:.6f}, {y:.6f}) outside "
                            f"bounds=({x_min:.6f}, {x_max:.6f}, "
                            f"{y_min:.6f}, {y_max:.6f})"
                        )
            return _collision_failure_reason(
                m, d, allowed_pairs,
                allowed_penetration=allowed_penetration,
                ignored_body_substrings=ignored_body_substrings,
                penetration_body_names=penetration_body_names,
            )

        def isValid(self, state) -> bool:
            return self.reason(state) is None

    checker = AttachedObjectChecker(si)
    ss.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(0.001)
    start, goal = space.allocState(), space.allocState()
    for i in range(ndim):
        start[i], goal[i] = float(q_start[i]), float(q_goal[i])
    ss.setStartAndGoalStates(start, goal)
    ss.setPlanner(og.RRTConnect(si))
    start_reason = checker.reason(start)
    goal_reason = checker.reason(goal)
    if start_reason is not None:
        print(f"[OMPL PREFLIGHT] start invalid: {start_reason}")
    if goal_reason is not None:
        print(f"[OMPL PREFLIGHT] goal invalid: {goal_reason}")
    if start_reason is not None or goal_reason is not None:
        return None
    if not ss.solve(solve_time):
        return None
    if hasattr(ss, "haveExactSolutionPath") and not ss.haveExactSolutionPath():
        print("[OMPL] approximate solution rejected")
        return None
    path = ss.getSolutionPath()
    simplifier = og.PathSimplifier(si)
    simplifier.simplifyMax(path)
    path.interpolate(interpolate_count)
    q_path = [np.array([path.getState(i)[j] for j in range(ndim)], dtype=float) for i in range(path.getStateCount())]
    if np.linalg.norm(q_path[-1] - q_goal) > 1e-3:
        print("[OMPL] endpoint mismatch rejected")
        return None
    return q_path



def move_to_grasp_pose(
    m,
    d,
    qpos_ids,
    dof_ids,
    limits,
    eef_body_name,
    source_T_world_object,
    T_object_eef,
    hover_offset=(0.0, 0.0, 0.10),
):
    """
    Solve IK for hover and grasp poses, move arm to grasp position.
    Returns q_hover and q_start (grasp configuration).
    """
    source_T = np.asarray(source_T_world_object, dtype=float)
    d_init = mujoco.MjData(m)
    mujoco.mj_forward(m, d_init)
    q_home = get_arm_q(d_init, qpos_ids)

    hover_offset = np.asarray(hover_offset, dtype=float)
    if hover_offset.shape == ():
        hover_offset = np.array([0.0, 0.0, float(hover_offset)])
    if hover_offset.shape != (3,):
        raise ValueError("hover_offset must be a scalar z offset or a 3-vector")

    # Hover pose: offset from the grasp location in the object's frame.
    T_hover = source_T @ T_object_eef
    T_hover[:3, 3] += source_T[:3, :3] @ hover_offset
    q_hover, ok_hover = solve_ik(
        m, d, qpos_ids, dof_ids, limits, eef_body_name, T_hover, q_home
    )
    if not ok_hover:
        return _result(
            "failed",
            "hover IK failed",
            q_hover=None,
            q_start=None,
        )

    # Grasp pose: contact with object
    T_grasp = source_T @ T_object_eef
    q_start, ok_grasp = solve_ik(
        m, d, qpos_ids, dof_ids, limits, eef_body_name, T_grasp, q_hover
    )
    if not ok_grasp:
        return _result(
            "failed",
            "grasp IK failed",
            q_hover=None,
            q_start=None,
        )

    return _result("success", q_hover=q_hover, q_start=q_start)


def grasp_and_attach_object(
    m,
    d,
    qpos_ids,
    eef_body_name,
    source_T_world_object,
    object_freejoint_name,
    close_ctrl_value=None,
    gripper_actuator_id=None,
):
    # Make sure current kinematics are valid at q_grasp
    mujoco.mj_forward(m, d)

    # Compute attachment immediately, before any mj_step can move the arm/object
    T_world_eef = get_body_transform(m, d, eef_body_name)
    T_eef_object = np.linalg.inv(T_world_eef) @ np.asarray(source_T_world_object, dtype=float)

    # Close only the gripper actuator, not all controls
    if close_ctrl_value is not None and gripper_actuator_id is not None:
        d.ctrl[int(gripper_actuator_id)] = float(close_ctrl_value)
        set_gripper_qpos_by_command(m, d, close_ctrl_value)
        mujoco.mj_forward(m, d)

    return _result("success", T_eef_object=T_eef_object)


def interpolate_attached_object_motion(
    m,
    d,
    q_start,
    qpos_ids,
    dof_ids,
    limits,
    eef_body_name,
    object_freejoint_name,
    T_eef_object,
    start_T_world_object,
    goal_T_world_object,
    allowed_contact_substrings,
    interpolate_count=30,
    ignored_body_substrings=(),
    allowed_penetration=0.003,
    penetration_body_names=(),
    goal_collision_waypoint_allowance=0,
    stop_short_waypoints=0,
):
    """IK-track a straight object-pose interpolation with quaternion slerp.

    stop_short_waypoints truncates the path before it reaches the goal pose,
    e.g. interpolate_count=30, stop_short_waypoints=3 stops at waypoint 27 —
    useful for releasing the object slightly above the goal and letting
    gravity settle the rest.
    """
    count = max(2, int(interpolate_count))
    stop_short = max(0, min(count - 1, int(stop_short_waypoints)))
    q_start = np.asarray(q_start, dtype=float)
    start_T = np.asarray(start_T_world_object, dtype=float)
    goal_T = np.asarray(goal_T_world_object, dtype=float)
    allowed_pairs = tuple(allowed_contact_substrings)
    ignored_bodies = tuple(ignored_body_substrings)
    penetration_bodies = tuple(penetration_body_names)
    waypoint_allowance = max(0, int(goal_collision_waypoint_allowance))
    T_object_eef = np.linalg.inv(np.asarray(T_eef_object, dtype=float))

    _apply_attached_state(
        m, d, q_start, qpos_ids, eef_body_name,
        object_freejoint_name, T_eef_object,
    )
    if not _collision_free(
        m,
        d,
        allowed_pairs,
        allowed_penetration=allowed_penetration,
        ignored_body_substrings=ignored_bodies,
        penetration_body_names=penetration_bodies,
    ):
        return _result("failed", "linear path start is in collision", q_path=None)

    q_path = [q_start.copy()]
    q_seed = q_start.copy()
    start_quat = mat_to_quat_wxyz(start_T[:3, :3])
    goal_quat = mat_to_quat_wxyz(goal_T[:3, :3])
    if np.dot(start_quat, goal_quat) < 0.0:
        goal_quat = -goal_quat
    quat_dot = float(np.clip(np.dot(start_quat, goal_quat), -1.0, 1.0))

    for index, alpha in enumerate(np.linspace(0.0, 1.0, count)[1 : count - stop_short], start=1):
        T_world_object = start_T.copy()
        T_world_object[:3, 3] = (
            (1.0 - alpha) * start_T[:3, 3]
            + alpha * goal_T[:3, 3]
        )
        if quat_dot > 0.9995:
            quat = start_quat + alpha * (goal_quat - start_quat)
            quat /= np.linalg.norm(quat)
        else:
            angle = np.arccos(quat_dot)
            quat = (
                np.sin((1.0 - alpha) * angle) * start_quat
                + np.sin(alpha * angle) * goal_quat
            ) / np.sin(angle)
        T_world_object[:3, :3] = quat_wxyz_to_mat(quat)
        T_world_eef = T_world_object @ T_object_eef
        q, ok = solve_ik(
            m,
            d,
            qpos_ids,
            dof_ids,
            limits,
            eef_body_name,
            T_world_eef,
            q_seed,
        )
        if not ok:
            _apply_attached_state(
                m, d, q_start, qpos_ids, eef_body_name,
                object_freejoint_name, T_eef_object,
            )
            return _result(
                "failed",
                f"linear motion IK failed at waypoint {index}/{count - 1}",
                q_path=None,
            )

        _apply_attached_state(
            m, d, q, qpos_ids, eef_body_name,
            object_freejoint_name, T_eef_object,
        )
        collision_free_now = _collision_free(
            m,
            d,
            allowed_pairs,
            allowed_penetration=allowed_penetration,
            ignored_body_substrings=ignored_bodies,
            penetration_body_names=penetration_bodies,
        )
        if not collision_free_now:
            waypoints_remaining = count - 1 - index
            if waypoints_remaining <= waypoint_allowance:
                print(
                    f"[LINEAR] tolerating collision at waypoint "
                    f"{index}/{count - 1}; {waypoints_remaining} waypoint(s) "
                    f"remain within allowance {waypoint_allowance}"
                )
                q_path.append(q.copy())
                q_seed = q
                continue
            _apply_attached_state(
                m, d, q_start, qpos_ids, eef_body_name,
                object_freejoint_name, T_eef_object,
            )
            return _result(
                "failed",
                f"linear motion collision at waypoint {index}/{count - 1}",
                q_path=None,
            )
        q_path.append(q.copy())
        q_seed = q

    return _result("success", q_path=q_path)

# Planning data and implementation helpers
PREINSERTION_INTERPOLATE_COUNT = 30
GRASP_INTERPOLATE_COUNT = 15
# Home plus two deterministic random starts balances alternate-branch coverage
# against the cost of checking multiple goal rotations for every grasp.
GOAL_IK_RANDOM_STARTS = 2
AXIS_VECTORS = {
    f"{'+' if sign > 0 else '-'}{axis}": sign * basis
    for axis, basis in zip("xyz", np.eye(3))
    for sign in (1, -1)
}


@dataclass
class PlanningContext:
    args: PipelineConfig
    model: mujoco.MjModel
    eval_data: mujoco.MjData
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    limits: np.ndarray
    arm_joint_names: tuple[str, ...]
    q_home: np.ndarray
    initial_qpos: np.ndarray
    initial_ctrl: np.ndarray
    gripper_info: dict
    gripper_open_ctrl: float
    allowed_contacts: tuple[tuple[str, str], ...]
    ignored_collision_bodies: tuple[str, ...]


@dataclass
class GraspPlan:
    candidate: GraspCandidate
    source_pose: ObjectPose
    goal_z_indices: tuple[int, ...]
    goal_q_by_index: tuple[tuple[int, np.ndarray], ...]
    move: dict
    home_to_pregrasp_q_path: list[np.ndarray]
    pregrasp_to_grasp_q_path: list[np.ndarray]

    @property
    def index(self) -> int:
        return self.candidate.index

    @property
    def gripper_close_ctrl(self) -> float:
        return self.candidate.close_ctrl_value

    @property
    def q_grasp(self) -> np.ndarray:
        return self.move["q_start"]

    @property
    def q_pregrasp(self) -> np.ndarray:
        return self.move["q_hover"]


@dataclass
class PreinsertionPlan:
    settings: dict
    position_range: np.ndarray
    rng: np.random.Generator


@dataclass
class GoalPlanCandidate:
    index: int
    goal_T: np.ndarray
    preinsertion_T: np.ndarray
    q_planning_goal: np.ndarray | None


@dataclass
class PlanningResult:
    source_pose: ObjectPose
    refined_goal_pose: ObjectPose
    preinsertion: dict
    trajectory: dict


def configured_preinsertion(args: PipelineConfig) -> PreinsertionPlan:
    preinsertion = {
        "use_preinsertion": bool(args.use_preinsertion),
        "axis": args.preinsertion_axis if args.use_preinsertion else None,
        "reasoning": (
            "Pre-insertion is controlled by benchmark config, not VLM. "
            f"use_preinsertion={bool(args.use_preinsertion)}, "
            f"axis={args.preinsertion_axis!r}."
        ),
    }
    if args.use_preinsertion and args.preinsertion_axis not in AXIS_VECTORS:
        raise ValueError(
            "preinsertion_axis must be one of +x, -x, +y, -y, +z, -z "
            f"when use_preinsertion is enabled; got {args.preinsertion_axis!r}"
        )
    return PreinsertionPlan(
        settings=preinsertion,
        position_range=np.asarray(args.preinsertion_position_range, dtype=float),
        rng=np.random.default_rng(0),
    )


def _make_context(args: PipelineConfig, xml_path: Path) -> PlanningContext:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    eval_data = mujoco.MjData(model)
    mujoco.mj_forward(model, eval_data)
    arm_joint_names = tuple(ARM_JOINT_NAMES.split(","))
    qpos_ids, dof_ids, limits = get_joint_info(model, arm_joint_names)
    gripper_info = get_gripper_control_info(model, GRIPPER_ACTUATOR_SUFFIX)
    gripper_open_ctrl = (
        float(OPEN_CTRL_VALUE)
        if OPEN_CTRL_VALUE is not None
        else float(gripper_info["open_ctrl"])
    )
    return PlanningContext(
        args=args,
        model=model,
        eval_data=eval_data,
        qpos_ids=qpos_ids,
        dof_ids=dof_ids,
        limits=limits,
        arm_joint_names=arm_joint_names,
        q_home=get_arm_q(eval_data, qpos_ids),
        initial_qpos=eval_data.qpos.copy(),
        initial_ctrl=eval_data.ctrl.copy(),
        gripper_info=gripper_info,
        gripper_open_ctrl=gripper_open_ctrl,
        allowed_contacts=tuple(tuple(x.split(":")) for x in args.allowed_contact_pair),
        ignored_collision_bodies=(),
    )


def _write_trajectory(path: Path, trajectory: dict) -> None:
    path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    reason = trajectory.get("reason") or "ok"
    print(
        f"[trajectory] {trajectory.get('status')} | {reason} | dir={path.parent}"
    )


def _log_if(condition: bool, message: str) -> None:
    if condition:
        print(message)


def _as_waypoint_list(q_path) -> list[list[float]]:
    return [np.asarray(q, dtype=float).tolist() for q in q_path]


def _move_phase(name: str, q_path, dt: float) -> dict:
    return {
        "type": "move",
        "name": name,
        "dt": dt,
        "waypoints": _as_waypoint_list(q_path),
    }


def _build_phase_waypoint_plan(
    *,
    ctx: PlanningContext,
    grasp_plan: GraspPlan,
    selected_plan: dict,
    selected_candidate: GoalPlanCandidate,
    pre: PreinsertionPlan,
    selected_T_eef_object: np.ndarray,
    source_xml: Path,
) -> dict:
    q_path = [np.asarray(q, dtype=float) for q in selected_plan["q_path"]]
    if not q_path:
        raise ValueError("selected_plan q_path must contain at least one waypoint.")

    use_pre = pre.settings["use_preinsertion"]

    if use_pre and selected_plan.get("ompl_q_path"):
        q_preinsertion = np.asarray(selected_plan["ompl_q_path"][-1], dtype=float)
        release_to_preinsertion_path = interpolate_joint_path(
            q_path[-1], q_preinsertion,
            steps_per_segment=STAGE_INTERPOLATE_COUNT,
        )
        q_after_release = q_preinsertion
    else:
        release_to_preinsertion_path = None
        q_after_release = q_path[-1]

    post_hover = ctx.args.post_release_hover_m
    if post_hover > 0.0:
        set_arm_q(ctx.eval_data, ctx.qpos_ids, q_after_release)
        mujoco.mj_forward(ctx.model, ctx.eval_data)
        T_release_eef = get_body_transform(ctx.model, ctx.eval_data, EEF_BODY_NAME)
        T_post_hover = T_release_eef.copy()
        T_post_hover[2, 3] += post_hover
        q_post_hover, ok_hover = solve_ik(
            ctx.model, ctx.eval_data, ctx.qpos_ids, ctx.dof_ids,
            ctx.limits, EEF_BODY_NAME, T_post_hover, q_after_release,
        )
        if ok_hover:
            post_hover_path = interpolate_joint_path(
                q_after_release, q_post_hover,
                steps_per_segment=STAGE_INTERPOLATE_COUNT,
            )
            return_home_q_path = interpolate_joint_path(
                q_post_hover, ctx.q_home,
                steps_per_segment=STAGE_INTERPOLATE_COUNT,
            )
        else:
            print("[build] post-release hover IK failed — skipping hover")
            post_hover_path = None
            return_home_q_path = interpolate_joint_path(
                q_after_release, ctx.q_home,
                steps_per_segment=STAGE_INTERPOLATE_COUNT,
            )
    else:
        post_hover_path = None
        return_home_q_path = interpolate_joint_path(
            q_after_release,
            ctx.q_home,
            steps_per_segment=STAGE_INTERPOLATE_COUNT,
        )

    default_dt = 0.08
    phases = [
        _move_phase("home_to_pregrasp", grasp_plan.home_to_pregrasp_q_path, default_dt),
        _move_phase("pregrasp_to_grasp", grasp_plan.pregrasp_to_grasp_q_path, default_dt),
        {"type": "gripper", "command": "close"},
        _move_phase("grasp_to_pregrasp", selected_plan["grasp_to_pregrasp_path"], default_dt),
        _move_phase("move_to_final_position", q_path, default_dt),
        {"type": "gripper", "command": "open"},
        *([_move_phase("release_to_preinsertion", release_to_preinsertion_path, default_dt)] if release_to_preinsertion_path else []),
        *([_move_phase("post_release_hover", post_hover_path, default_dt)] if post_hover_path else []),
        _move_phase("return_home", return_home_q_path, default_dt),
    ]
    metadata = {
        "benchmark": ctx.args.benchmark,
        "source_xml": str(source_xml),
        "selected_grasp_candidate": grasp_plan.index,
        "selected_grasp_key": grasp_plan.candidate.key,
        "selected_grasp_source": grasp_plan.candidate.source,
        "selected_grasp_score": grasp_plan.candidate.score,
        "selected_grasp_width_m": grasp_plan.candidate.detected_width_m,
        "selected_goal_z_sample": selected_candidate.index,
        "selected_preinsertion_position_sample": selected_plan["preinsertion_sample"],
        "planner": {
            "type": "ompl_then_linear_insertion" if use_pre else "ompl_attached_rrtconnect",
            "solve_time": OMPL_SOLVE_TIME,
            "interpolate_count": OMPL_INTERPOLATE_COUNT,
            "raw_q_path": _as_waypoint_list(selected_plan["ompl_q_path"]),
            "linear_q_path": (
                _as_waypoint_list(selected_plan["linear_q_path"])
                if selected_plan["linear_q_path"] is not None
                else []
            ),
            "linear_interpolate_count": PREINSERTION_INTERPOLATE_COUNT if use_pre else 0,
        },
        "preinsertion": pre.settings,
        "mujoco": {
            "xml_path": str(source_xml),
            "object_freejoint_name": ctx.args.object_freejoint_name,
            "eef_body_name": EEF_BODY_NAME,
            "gripper_actuator": ctx.gripper_info,
            "gripper": {
                "open_ctrl": ctx.gripper_open_ctrl,
                "close_ctrl": grasp_plan.gripper_close_ctrl,
            },
            "T_eef_object": np.asarray(selected_T_eef_object, dtype=float).tolist(),
            "source_T_world_object": np.asarray(
                grasp_plan.source_pose.T_world_object, dtype=float
            ).tolist(),
        },
    }
    for phase in phases:
        if phase["type"] == "move":
            metadata[f"{phase['name']}_waypoint_count"] = len(phase["waypoints"])

    if use_pre:
        metadata["q_preinsertion"] = selected_plan["ompl_q_path"][-1].tolist()

    return {
        "status": "success",
        "format": "iiwa_phase_waypoint_plan_v1",
        "description": (
            "Phase-based plan exported from pipeline motion planning. Move phases "
            "contain MuJoCo-planned iiwa joint waypoints; gripper phases hold the "
            "current arm command while the gripper acts."
        ),
        "dt": default_dt,
        "units": "radians",
        "arm_joint_names": list(ctx.arm_joint_names),
        "phases": phases,
        "metadata": metadata,
    }


def _validate_controller_trajectory(trajectory: dict) -> None:
    """Check the phase artifact before handing it to replay/controller code."""
    if trajectory.get("status") != "success":
        return
    if trajectory.get("format") != "iiwa_phase_waypoint_plan_v1":
        raise ValueError("Successful trajectory has unexpected format.")

    joint_count = len(trajectory.get("arm_joint_names", []))
    if joint_count <= 0:
        raise ValueError("Trajectory has no arm_joint_names.")
    phases = trajectory.get("phases", [])
    if not phases:
        raise ValueError("Successful trajectory has no phases.")

    gripper_commands = []
    for phase_index, phase in enumerate(phases):
        typ = phase.get("type")
        if typ == "move":
            waypoints = np.asarray(phase.get("waypoints"), dtype=float)
            name = phase.get("name", phase_index)
            if waypoints.ndim != 2 or waypoints.shape[1] != joint_count or len(waypoints) == 0:
                raise ValueError(f"Move phase {name!r} must have shape (N, {joint_count}).")
            if not np.all(np.isfinite(waypoints)):
                raise ValueError(f"Move phase {name!r} has non-finite waypoints.")
            if float(phase.get("dt", trajectory.get("dt", 0.0))) <= 0:
                raise ValueError(f"Move phase {name!r} has non-positive dt.")
        elif typ == "gripper":
            command = phase.get("command")
            if command not in {"open", "close"}:
                raise ValueError(f"Invalid gripper command: {command!r}.")
            gripper_commands.append(command)
        else:
            raise ValueError(f"Unknown phase type: {typ!r}.")
    if gripper_commands != ["close", "open"]:
        raise ValueError("Trajectory must close then open the gripper exactly once.")

    mujoco_meta = trajectory.get("metadata", {}).get("mujoco", {})
    for key in ("T_eef_object", "source_T_world_object"):
        T = np.asarray(mujoco_meta.get(key), dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError(f"Trajectory {key} must be a finite 4x4 transform.")


def apply_trajectory_waypoint(
    m,
    d,
    waypoint,
    *,
    qpos_ids,
    gripper_actuator_id=None,
    gripper_open_ctrl=0.0,
    gripper_close_ctrl=0.0,
) -> None:
    """Apply arm joint positions and gripper command from a phase waypoint."""
    q = np.asarray(waypoint["arm_joint_positions"], dtype=float)
    if "gripper_command" in waypoint:
        gripper_command = float(waypoint["gripper_command"])
    else:
        gripper_state = waypoint.get("gripper_state", "open")
        gripper_command = gripper_close_ctrl if gripper_state == "closed" else gripper_open_ctrl
    set_arm_q(d, qpos_ids, q)
    set_gripper_command(m, d, gripper_actuator_id, gripper_command)
    set_gripper_qpos_by_command(m, d, gripper_command)
    mujoco.mj_forward(m, d)


def _phase_move_waypoints(trajectory: dict) -> list[dict]:
    """Flatten phase moves for MuJoCo replay while preserving gripper events."""
    flat = []
    gripper_state = "open"
    for phase in trajectory.get("phases", []):
        if phase.get("type") == "move":
            flat.extend(
                {"arm_joint_positions": q, "gripper_state": gripper_state}
                for q in phase.get("waypoints", [])
            )
        elif phase.get("type") == "gripper":
            command = phase.get("command")
            event_by_command = {
                "close": ("closed", "attach_object"),
                "open": ("open", "release_object"),
            }
            if command not in event_by_command:
                raise ValueError(f"Unknown gripper command: {command!r}")
            gripper_state, event = event_by_command[command]
            if flat:
                flat.append({**flat[-1], "gripper_state": gripper_state, "event": event})
        else:
            raise ValueError(f"Unknown phase type: {phase.get('type')!r}")
    return flat


def _place_replay_marker(m, d, body_name: str, T_world_body: np.ndarray) -> None:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    T_world_body = np.asarray(T_world_body, dtype=float)
    mocap_id = int(m.body_mocapid[bid])
    if mocap_id >= 0:
        d.mocap_pos[mocap_id] = T_world_body[:3, 3]
        d.mocap_quat[mocap_id] = mat_to_quat_wxyz(T_world_body[:3, :3])
        return
    for joint_id in range(m.njnt):
        if m.jnt_bodyid[joint_id] != bid or m.jnt_type[joint_id] != 0:
            continue
        qpos_adr = int(m.jnt_qposadr[joint_id])
        d.qpos[qpos_adr:qpos_adr + 3] = T_world_body[:3, 3]
        d.qpos[qpos_adr + 3:qpos_adr + 7] = mat_to_quat_wxyz(T_world_body[:3, :3])
        break


def replay_pick_place_trajectory(
    *,
    xml_path: Path,
    trajectory: dict,
    object_freejoint_name: str,
    goal_marker_body: str = "",
    goal_T_world_object: np.ndarray | None = None,
    seconds_per_waypoint: float = 0.03,
) -> None:
    """Loop an iiwa_phase_waypoint_plan_v1 trajectory in a MuJoCo passive viewer."""
    import mujoco.viewer

    _validate_controller_trajectory(trajectory)
    if trajectory.get("status") != "success":
        return

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    arm_joint_names = tuple(trajectory.get("arm_joint_names") or ARM_JOINT_NAMES.split(","))
    qpos_ids, _, _ = get_joint_info(model, arm_joint_names)
    gripper_info = get_gripper_control_info(model, GRIPPER_ACTUATOR_SUFFIX)
    mujoco_meta = trajectory.get("metadata", {}).get("mujoco", {})
    gripper = mujoco_meta.get("gripper", {})
    eef_body_name = str(mujoco_meta.get("eef_body_name", EEF_BODY_NAME))
    T_eef_object = np.asarray(mujoco_meta["T_eef_object"], dtype=float)
    source_T_world_object = np.asarray(mujoco_meta["source_T_world_object"], dtype=float)
    waypoints = _phase_move_waypoints(trajectory)
    if not waypoints:
        raise ValueError("Trajectory has no waypoints.")

    set_freejoint_pose_by_name(model, data, object_freejoint_name, source_T_world_object)
    if goal_marker_body and goal_T_world_object is not None:
        _place_replay_marker(model, data, goal_marker_body, goal_T_world_object)
    mujoco.mj_forward(model, data)

    print("[replay] looping planned motion in MuJoCo viewer; close the viewer to continue")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        index = 0
        last = 0.0
        attached_to_eef = False
        current_object_T = source_T_world_object.copy()
        open_ctrl = float(gripper.get("open_ctrl", gripper_info["open_ctrl"]))
        close_ctrl = float(gripper.get("close_ctrl", gripper_info["close_ctrl"]))

        while viewer.is_running():
            now = time.time()
            if now - last >= seconds_per_waypoint:
                waypoint = waypoints[index]
                event = waypoint.get("event")
                if event == "attach_object":
                    attached_to_eef = True
                elif event == "release_object":
                    T_world_eef = get_body_transform(model, data, eef_body_name)
                    current_object_T = T_world_eef @ T_eef_object
                    attached_to_eef = False

                apply_trajectory_waypoint(
                    model,
                    data,
                    waypoint,
                    qpos_ids=qpos_ids,
                    gripper_actuator_id=gripper_info["actuator_id"],
                    gripper_open_ctrl=open_ctrl,
                    gripper_close_ctrl=close_ctrl,
                )
                if attached_to_eef:
                    T_world_eef = get_body_transform(model, data, eef_body_name)
                    set_freejoint_pose_by_name(
                        model, data, object_freejoint_name, T_world_eef @ T_eef_object
                    )
                else:
                    set_freejoint_pose_by_name(
                        model, data, object_freejoint_name, current_object_T
                    )
                mujoco.mj_forward(model, data)

                index += 1
                if index >= len(waypoints):
                    index = 0
                    attached_to_eef = False
                    current_object_T = source_T_world_object.copy()
                    set_freejoint_pose_by_name(
                        model, data, object_freejoint_name, current_object_T
                    )
                    mujoco.mj_forward(model, data)
                last = now

            viewer.sync()
            time.sleep(0.001)


def _plan_fixed_object_path(
    ctx: PlanningContext,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    fixed_object_T: np.ndarray,
    interpolate_count: int | None = None,
) -> list[np.ndarray] | None:
    args = ctx.args
    plan_data = mujoco.MjData(ctx.model)
    plan_data.qpos[:] = ctx.eval_data.qpos
    plan_data.qvel[:] = ctx.eval_data.qvel
    set_freejoint_pose_by_name(
        ctx.model, plan_data, args.object_freejoint_name, fixed_object_T
    )
    mujoco.mj_forward(ctx.model, plan_data)
    return plan_ompl_motion(
        ctx.model,
        plan_data,
        q_start,
        q_goal,
        ctx.qpos_ids,
        ctx.limits,
        eef_body_name=None,
        object_freejoint_name=args.object_freejoint_name,
        T_eef_object=None,
        allowed_pairs=ctx.allowed_contacts,
        interpolate_count=interpolate_count if interpolate_count is not None else STAGE_INTERPOLATE_COUNT,
        ignored_body_substrings=ctx.ignored_collision_bodies,
        allowed_penetration=args.allowed_penetration,
        penetration_body_names=(args.mujoco_active_body,),
        fixed_object_T=fixed_object_T,
    )


def _goal_allowed_contacts(ctx: PlanningContext) -> tuple[tuple[str, str], ...]:
    """Allow only the intended gripper/object contacts at an exact goal."""
    active_body = ctx.args.mujoco_active_body
    broad_arm_pair = ("arm1_", active_body)
    return tuple(
        pair for pair in ctx.allowed_contacts if tuple(pair) != broad_arm_pair
    ) + (
        ("arm1_finger_1", active_body),
        ("arm1_finger_2", active_body),
        (EEF_BODY_NAME, active_body),
    )


def _goal_configuration_check(
    ctx: PlanningContext,
    candidate: GraspCandidate,
    goal_T_world_object: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Return the first collision-free goal IK branch and any rejection reason."""
    args = ctx.args
    target_T_world_eef = (
        np.asarray(goal_T_world_object, dtype=float) @ candidate.T_object_eef
    )
    rng = np.random.default_rng(0)
    seeds = [np.asarray(ctx.q_home, dtype=float).copy()]
    seeds.extend(
        rng.uniform(ctx.limits[:, 0], ctx.limits[:, 1])
        for _ in range(GOAL_IK_RANDOM_STARTS)
    )
    first_collision = None
    ik_successes = 0

    for q_seed in seeds:
        data = mujoco.MjData(ctx.model)
        data.qpos[:] = ctx.initial_qpos
        data.qvel[:] = 0.0
        if ctx.model.nu:
            data.ctrl[:] = ctx.initial_ctrl
        set_gripper_qpos_by_command(
            ctx.model, data, candidate.close_ctrl_value
        )
        mujoco.mj_forward(ctx.model, data)

        q_goal, ok = solve_ik(
            ctx.model,
            data,
            ctx.qpos_ids,
            ctx.dof_ids,
            ctx.limits,
            EEF_BODY_NAME,
            target_T_world_eef,
            q_seed,
        )
        if not ok:
            continue
        ik_successes += 1

        _apply_attached_state(
            ctx.model,
            data,
            q_goal,
            ctx.qpos_ids,
            EEF_BODY_NAME,
            args.object_freejoint_name,
            np.linalg.inv(candidate.T_object_eef),
        )
        reason = _collision_failure_reason(
            ctx.model,
            data,
            _goal_allowed_contacts(ctx),
            allowed_penetration=args.allowed_penetration,
            ignored_body_substrings=ctx.ignored_collision_bodies,
            penetration_body_names=(args.mujoco_active_body,),
        )
        if reason is None:
            return q_goal.copy(), None
        if first_collision is None:
            first_collision = reason

    if first_collision is not None:
        return None, (
            f"{first_collision}; all {ik_successes} solved IK branch(es) collide"
        )
    return None, f"goal IK failed after {len(seeds)} deterministic starts"


def _filter_grasps_for_goal(
    ctx: PlanningContext,
    candidates: Sequence[GraspCandidate],
    refined_goal_pose: ObjectPose,
    *,
    max_results: int | None = None,
) -> list[GoalFilteredGrasp]:
    """Reject a grasp before source planning unless a goal-Z pose is valid."""
    goal_poses = sample_equivalent_poses_about_z(
        refined_goal_pose.T_world_object,
        ctx.args.z_rotation_samples,
    )
    accepted: list[GoalFilteredGrasp] = []
    for candidate in candidates:
        valid_indices = []
        goal_q_by_index = []
        failures = []
        for goal_index, goal_T in enumerate(goal_poses):
            q_goal, reason = _goal_configuration_check(ctx, candidate, goal_T)
            if reason is None and q_goal is not None:
                valid_indices.append(goal_index)
                goal_q_by_index.append((goal_index, q_goal.copy()))
            else:
                failures.append(reason or "goal IK did not return a configuration")
        if not valid_indices:
            summary = failures[0] if failures and len(set(failures)) == 1 else "all goal poses invalid"
            print(
                f"[planning:grasp] {candidate.key}: rejected at goal ({summary})"
            )
            continue
        if len(valid_indices) != len(goal_poses):
            print(
                f"[planning:grasp] {candidate.key}: "
                f"{len(valid_indices)}/{len(goal_poses)} goal Z pose(s) collision-free"
            )
        accepted.append(
            GoalFilteredGrasp(
                candidate,
                tuple(valid_indices),
                tuple(goal_q_by_index),
            )
        )
        if max_results is not None and len(accepted) >= int(max_results):
            break
    return accepted


def _iter_grasp_plans(
    ctx: PlanningContext,
    source_pose: ObjectPose,
    grasps: Sequence[GoalFilteredGrasp],
):
    args = ctx.args
    multiple_grasps = len(grasps) > 1 or any(
        grasp.candidate.expand_source_z and args.z_rotation_samples > 1
        for grasp in grasps
    )
    _log_if(
        multiple_grasps,
        f"[planning:grasp] planning {len(grasps)} goal-feasible grasp(s)",
    )
    for prepared in grasps:
        candidate = prepared.candidate
        source_candidates = (
            sample_equivalent_poses_about_z(
                source_pose.T_world_object, args.z_rotation_samples
            )
            if candidate.expand_source_z
            else [np.asarray(source_pose.T_world_object, dtype=float).copy()]
        )
        _n_ik_fail = _n_hover_fail = _n_path_fail = 0
        for z_index, source_T in enumerate(source_candidates):
            # Each candidate starts from the same open/home state. In
            # particular, do not inherit a closed gripper from a failed
            # AnyGrasp attempt before entering the authored fallback batch.
            ctx.eval_data.qpos[:] = ctx.initial_qpos
            ctx.eval_data.qvel[:] = 0.0
            if ctx.model.nu:
                ctx.eval_data.ctrl[:] = ctx.initial_ctrl
            set_arm_q(ctx.eval_data, ctx.qpos_ids, ctx.q_home)
            set_freejoint_pose_by_name(
                ctx.model, ctx.eval_data, args.object_freejoint_name, source_T
            )
            mujoco.mj_forward(ctx.model, ctx.eval_data)
            candidate_move = move_to_grasp_pose(
                ctx.model,
                ctx.eval_data,
                ctx.qpos_ids,
                ctx.dof_ids,
                ctx.limits,
                EEF_BODY_NAME,
                source_T,
                candidate.T_object_eef,
                candidate.hover_offset_object_xyz,
            )
            if candidate_move["status"] != "success":
                _n_ik_fail += 1
                continue

            path_to_hover = _plan_fixed_object_path(
                ctx, ctx.q_home, candidate_move["q_hover"], source_T
            )
            if path_to_hover is None:
                _n_hover_fail += 1
                continue

            path_to_grasp = _plan_fixed_object_path(
                ctx, candidate_move["q_hover"], candidate_move["q_start"], source_T,
                interpolate_count=GRASP_INTERPOLATE_COUNT,
            )
            if path_to_grasp is None:
                _n_path_fail += 1
                continue

            selected_source_pose = ObjectPose(
                source_pose.name,
                source_T,
                source_pose.confidence,
                (
                    f"{source_pose.note}; {candidate.key}; "
                    f"equivalent Z rotation sample {z_index}"
                ),
            )
            _log_if(multiple_grasps, f"[planning:grasp] {candidate.key}, source Z {z_index}: feasible"
                    f" ({_n_ik_fail} IK, {_n_hover_fail} hover, {_n_path_fail} path failures before)")
            yield GraspPlan(
                candidate=candidate,
                source_pose=selected_source_pose,
                goal_z_indices=prepared.goal_z_indices,
                goal_q_by_index=prepared.goal_q_by_index,
                move=candidate_move,
                home_to_pregrasp_q_path=path_to_hover,
                pregrasp_to_grasp_q_path=path_to_grasp,
            )


def _measure_attachment(ctx: PlanningContext, grasp_plan: GraspPlan) -> np.ndarray | None:
    args = ctx.args
    set_arm_q(ctx.eval_data, ctx.qpos_ids, grasp_plan.q_grasp)
    set_freejoint_pose_by_name(
        ctx.model,
        ctx.eval_data,
        args.object_freejoint_name,
        grasp_plan.source_pose.T_world_object,
    )
    mujoco.mj_forward(ctx.model, ctx.eval_data)
    grasp_result = grasp_and_attach_object(
        ctx.model,
        ctx.eval_data,
        ctx.qpos_ids,
        EEF_BODY_NAME,
        grasp_plan.source_pose.T_world_object,
        args.object_freejoint_name,
        grasp_plan.gripper_close_ctrl,
        gripper_actuator_id=ctx.gripper_info["actuator_id"],
    )
    if grasp_result["status"] != "success":
        print(f"[planning:goal] grasping failed: {grasp_result.get('reason', 'unknown error')}")
        return None
    return grasp_result["T_eef_object"]


def _goal_plan_candidates(
    ctx: PlanningContext,
    refined_goal_pose: ObjectPose,
    pre: PreinsertionPlan,
    T_object_eef: np.ndarray,
    q_start: np.ndarray,
    allowed_goal_z_indices: Sequence[int] | None = None,
    checked_goal_q_by_index: Sequence[tuple[int, np.ndarray]] = (),
) -> list[GoalPlanCandidate]:
    args = ctx.args
    preinsertion = pre.settings
    offset_world = (
        args.preinsertion_offset_m * AXIS_VECTORS[preinsertion["axis"]]
        if preinsertion["use_preinsertion"]
        else np.zeros(3)
    )
    candidates = []
    goal_candidates = sample_equivalent_poses_about_z(
        refined_goal_pose.T_world_object, args.z_rotation_samples
    )
    allowed_indices = (
        set(range(len(goal_candidates)))
        if allowed_goal_z_indices is None
        else {int(index) for index in allowed_goal_z_indices}
    )
    checked_goal_q = {
        int(index): np.asarray(q, dtype=float).copy()
        for index, q in checked_goal_q_by_index
    }
    T_eef_object = np.linalg.inv(np.asarray(T_object_eef, dtype=float))
    multiple_goals = len(goal_candidates) > 1
    _n_ik_ok = _n_ik_fail = _n_collision_fail = 0
    for index, goal_T in enumerate(goal_candidates):
        if index not in allowed_indices:
            continue
        pre_T = goal_T.copy()
        pre_T[:3, 3] += offset_world
        q_seed = checked_goal_q.get(index, q_start)
        candidate_q, ok_goal = _solve_eef_ik(
            ctx,
            goal_T @ T_object_eef,
            q_seed,
        )
        if not ok_goal:
            _n_ik_fail += 1
            continue

        _apply_attached_state(
            ctx.model,
            ctx.eval_data,
            candidate_q,
            ctx.qpos_ids,
            EEF_BODY_NAME,
            args.object_freejoint_name,
            T_eef_object,
        )
        collision_reason = _collision_failure_reason(
            ctx.model,
            ctx.eval_data,
            _goal_allowed_contacts(ctx),
            allowed_penetration=args.allowed_penetration,
            ignored_body_substrings=ctx.ignored_collision_bodies,
            penetration_body_names=(args.mujoco_active_body,),
        )
        if collision_reason is not None:
            _n_collision_fail += 1
            continue

        _n_ik_ok += 1
        candidates.append(GoalPlanCandidate(index, goal_T, pre_T, candidate_q))
    if multiple_goals:
        if preinsertion["use_preinsertion"]:
            print(
                f"[planning:goal] {len(candidates)} goal Z samples queued "
                "for pre-insertion after exact-goal collision filtering"
            )
        else:
            checked_count = _n_ik_ok + _n_ik_fail + _n_collision_fail
            print(
                f"[planning:goal] {_n_ik_ok}/{checked_count} goal Z samples "
                "have collision-free IK"
            )
    return candidates


def _solve_eef_ik(ctx: PlanningContext, target_T: np.ndarray, q_seed: np.ndarray):
    set_arm_q(ctx.eval_data, ctx.qpos_ids, q_seed)
    mujoco.mj_forward(ctx.model, ctx.eval_data)
    return solve_ik(
        ctx.model,
        ctx.eval_data,
        ctx.qpos_ids,
        ctx.dof_ids,
        ctx.limits,
        EEF_BODY_NAME,
        target_T,
        q_seed,
    )


def _plan_attached_to_q(
    ctx: PlanningContext,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    T_eef_object: np.ndarray,
    interpolate_count: int | None = None,
) -> list[np.ndarray] | None:
    args = ctx.args
    _apply_attached_state(
        ctx.model,
        ctx.eval_data,
        q_start,
        ctx.qpos_ids,
        EEF_BODY_NAME,
        args.object_freejoint_name,
        T_eef_object,
    )
    return plan_ompl_motion(
        ctx.model,
        ctx.eval_data,
        q_start,
        q_goal,
        ctx.qpos_ids,
        ctx.limits,
        EEF_BODY_NAME,
        args.object_freejoint_name,
        T_eef_object,
        ctx.allowed_contacts,
        solve_time=OMPL_SOLVE_TIME,
        interpolate_count=interpolate_count if interpolate_count is not None else OMPL_INTERPOLATE_COUNT,
        ignored_body_substrings=ctx.ignored_collision_bodies,
        allowed_penetration=args.allowed_penetration,
        penetration_body_names=(args.mujoco_active_body,),
    )


def _sample_preinsertion_path(
    ctx: PlanningContext,
    candidate: GoalPlanCandidate,
    pre: PreinsertionPlan,
    q_start: np.ndarray,
    T_eef_object: np.ndarray,
    T_object_eef: np.ndarray,
) -> dict | None:
    args = ctx.args
    sample_count = max(1, int(args.preinsertion_position_samples))
    consecutive_ompl_fails = 0
    _n_ik_fail = _n_ompl_fail = _n_linear_fail = 0
    for sample_index in range(sample_count):
        sample_pre_T = candidate.preinsertion_T.copy()
        sample_noise = np.zeros(3)
        if sample_index > 0:
            sample_noise = pre.rng.uniform(-pre.position_range, pre.position_range)
            sample_pre_T[:3, 3] += sample_noise

        sample_q, ok_pre = _solve_eef_ik(ctx, sample_pre_T @ T_object_eef, q_start)
        if not ok_pre:
            _n_ik_fail += 1
            continue

        ompl_q_path = _plan_attached_to_q(ctx, q_start, sample_q, T_eef_object)
        if ompl_q_path is None:
            consecutive_ompl_fails += 1
            _n_ompl_fail += 1
            if consecutive_ompl_fails >= 5:
                _log_if(
                    sample_count > 1,
                    f"[planning:select] goal Z sample {candidate.index}: aborting after {consecutive_ompl_fails} consecutive OMPL failures",
                )
                break
            continue
        consecutive_ompl_fails = 0

        linear_result = interpolate_attached_object_motion(
            ctx.model,
            ctx.eval_data,
            ompl_q_path[-1],
            ctx.qpos_ids,
            ctx.dof_ids,
            ctx.limits,
            EEF_BODY_NAME,
            args.object_freejoint_name,
            T_eef_object,
            sample_pre_T,
            candidate.goal_T,
            ctx.allowed_contacts,
            interpolate_count=PREINSERTION_INTERPOLATE_COUNT,
            ignored_body_substrings=ctx.ignored_collision_bodies,
            allowed_penetration=args.preinsertion_allowed_penetration,
            penetration_body_names=(args.mujoco_active_body,),
            goal_collision_waypoint_allowance=args.preinsertion_waypoint_allowance,
            stop_short_waypoints=args.preinsertion_stop_short_waypoints,
        )
        if linear_result["status"] != "success":
            _n_linear_fail += 1
            continue

        _log_if(
            sample_count > 1,
            f"[10e/10] goal Z sample {candidate.index}: pre-insertion found at sample {sample_index}/{sample_count - 1}"
            f" (IK={_n_ik_fail} OMPL={_n_ompl_fail} linear={_n_linear_fail} failures)",
        )
        linear_q_path = linear_result["q_path"]
        return {
            "q_path": ompl_q_path + linear_q_path[1:],
            "ompl_q_path": ompl_q_path,
            "linear_q_path": linear_q_path,
            "preinsertion_T": sample_pre_T,
            "preinsertion_sample": sample_index,
            "preinsertion_noise": sample_noise,
        }
    if sample_count > 1:
        print(f"[planning:select] goal Z sample {candidate.index}: all {sample_count} pre-insertion samples failed"
              f" (IK={_n_ik_fail} OMPL={_n_ompl_fail} linear={_n_linear_fail})")
    return None


def _plan_lift(
    ctx: PlanningContext,
    q_start: np.ndarray,
    T_eef_object: np.ndarray,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Plan a straight world-Z lift from the grasp pose. Returns (q_lift, q_path) or (None, [])."""
    lift_height = LIFT_HEIGHT_M
    if lift_height <= 0.0:
        return None, []

    _apply_attached_state(
        ctx.model, ctx.eval_data, q_start,
        ctx.qpos_ids, EEF_BODY_NAME,
        ctx.args.object_freejoint_name, T_eef_object,
    )
    T_world_eef = get_body_transform(ctx.model, ctx.eval_data, EEF_BODY_NAME)
    T_lift = T_world_eef.copy()
    T_lift[2, 3] += lift_height

    q_lift, ok = solve_ik(
        ctx.model, ctx.eval_data, ctx.qpos_ids, ctx.dof_ids,
        ctx.limits, EEF_BODY_NAME, T_lift, q_start,
    )
    if not ok:
        print("[planning] lift IK failed — skipping lift waypoint")
        return None, []

    lift_path = _plan_attached_to_q(ctx, q_start, q_lift, T_eef_object)
    if lift_path is None:
        print("[planning] lift OMPL failed — skipping lift waypoint")
        return None, []

    return q_lift, lift_path


def _plan_goal_candidate(
    ctx: PlanningContext,
    candidate: GoalPlanCandidate,
    pre: PreinsertionPlan,
    q_start: np.ndarray,
    T_eef_object: np.ndarray,
    T_object_eef: np.ndarray,
    goal_candidate_count: int,
) -> dict | None:
    multiple_goals = goal_candidate_count > 1
    if pre.settings["use_preinsertion"]:
        return _sample_preinsertion_path(
            ctx, candidate, pre, q_start, T_eef_object, T_object_eef
        )

    q_lifted, lift_path = _plan_lift(ctx, q_start, T_eef_object)
    q_for_goal = q_lifted if q_lifted is not None else q_start

    q_path = _plan_attached_to_q(ctx, q_for_goal, candidate.q_planning_goal, T_eef_object)
    if q_path is None:
        print(f"[planning:select] goal Z sample {candidate.index}: OMPL failed")
        return None

    full_path = lift_path + q_path if lift_path else q_path
    return {
        "q_path": full_path,
        "ompl_q_path": full_path,
        "linear_q_path": None,
        "preinsertion_T": candidate.preinsertion_T,
        "preinsertion_sample": None,
        "preinsertion_noise": np.zeros(3),
    }



def _log_selected_poses(
    goal_T: np.ndarray,
    passive_pose,
    grasp_T: np.ndarray,
    preinsertion_T: np.ndarray,
) -> None:
    def _fmt(T):
        xyz = np.round(T[:3, 3], 4)
        R = T[:3, :3]
        pitch = float(np.arcsin(-R[2, 0]))
        if abs(np.cos(pitch)) > 1e-6:
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
        else:
            roll = float(np.arctan2(-R[1, 2], R[1, 1]))
            yaw  = 0.0
        rpy_deg = np.round(np.degrees([roll, pitch, yaw]), 2)
        quat = np.round(mat_to_quat_wxyz(R), 4)
        return f"xyz={xyz.tolist()}  rpy_deg={rpy_deg.tolist()}  quat(wxyz)={quat.tolist()}"

    print(f"[planning] grasp pose (source active):  {_fmt(grasp_T)}")
    print(f"[planning] preinsertion pose:            {_fmt(preinsertion_T)}")
    print(f"[planning] selected active goal pose:   {_fmt(goal_T)}")
    if passive_pose is not None:
        print(f"[planning] passive object pose:          {_fmt(passive_pose.T_world_object)}")
    else:
        print("[planning] passive object pose:          None")

# ---------------------------------------------------------------------------
# Public planning interfaces
# ---------------------------------------------------------------------------

class IKSolver:
    """Robot kinematics boundary. Replace this class to swap IK implementations."""

    def __init__(self, context: PlanningContext):
        self.context = context

    def solve(self, target_T: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, bool]:
        ctx = self.context
        return solve_ik(
            ctx.model, ctx.eval_data, ctx.qpos_ids, ctx.dof_ids, ctx.limits,
            EEF_BODY_NAME, target_T, seed,
        )


class CollisionChecker:
    """Collision-policy boundary shared by free and attached-object planning."""

    def __init__(self, context: PlanningContext):
        self.context = context

    def state_is_valid(self, data: mujoco.MjData) -> bool:
        ctx = self.context
        return collision_free(
            ctx.model,
            data,
            allowed_contact_substrings=ctx.allowed_contacts,
            ignored_body_substrings=ctx.ignored_collision_bodies,
            allowed_penetration=ctx.args.allowed_penetration,
            penetration_body_names=(ctx.args.mujoco_active_body,),
            debug=False,
        )

    def plan_fixed_object(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        fixed_object_T: np.ndarray,
        *,
        interpolate_count: int | None = None,
    ) -> list[np.ndarray] | None:
        return _plan_fixed_object_path(
            self.context, q_start, q_goal, fixed_object_T, interpolate_count
        )


class GraspApproachPlanner:
    """Goal-filter candidates and plan collision-free source approaches."""

    def __init__(self, context: PlanningContext, collision: CollisionChecker):
        self.context = context
        self.collision = collision

    def filter_for_goal(
        self,
        candidates: Sequence[GraspCandidate],
        goal_pose: ObjectPose,
        *,
        max_results: int | None = None,
    ) -> list[GoalFilteredGrasp]:
        return _filter_grasps_for_goal(
            self.context,
            candidates,
            goal_pose,
            max_results=max_results,
        )

    def plan_approaches(
        self,
        source_pose: ObjectPose,
        grasps: Sequence[GoalFilteredGrasp],
    ) -> Iterator[GraspPlan]:
        yield from _iter_grasp_plans(self.context, source_pose, grasps)

    def attach(self, grasp: GraspPlan) -> np.ndarray | None:
        return _measure_attachment(self.context, grasp)


class AttachedObjectPlanner:
    """Plan motions while the object is rigidly attached to the end effector."""

    def __init__(
        self,
        context: PlanningContext,
        ik: IKSolver,
        collision: CollisionChecker,
    ):
        self.context = context
        self.ik = ik
        self.collision = collision

    def plan_to_configuration(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        T_eef_object: np.ndarray,
        *,
        interpolate_count: int | None = None,
    ) -> list[np.ndarray] | None:
        return _plan_attached_to_q(
            self.context,
            q_start,
            q_goal,
            T_eef_object,
            interpolate_count=interpolate_count,
        )

    def goal_candidates(
        self,
        goal_pose: ObjectPose,
        preinsertion: PreinsertionPlan,
        T_object_eef: np.ndarray,
        q_seed: np.ndarray,
        allowed_goal_z_indices: Sequence[int] | None = None,
        checked_goal_q_by_index: Sequence[tuple[int, np.ndarray]] = (),
    ) -> list[GoalPlanCandidate]:
        return _goal_plan_candidates(
            self.context,
            goal_pose,
            preinsertion,
            T_object_eef,
            q_seed,
            allowed_goal_z_indices,
            checked_goal_q_by_index,
        )

    def plan_lift(
        self,
        q_start: np.ndarray,
        T_eef_object: np.ndarray,
    ) -> tuple[np.ndarray | None, list[np.ndarray]]:
        return _plan_lift(self.context, q_start, T_eef_object)


class InsertionPlanner:
    """Plan pre-insertion sampling and the final straight insertion segment."""

    def __init__(
        self,
        context: PlanningContext,
        attached: AttachedObjectPlanner,
    ):
        self.context = context
        self.attached = attached

    @staticmethod
    def configure(args: PipelineConfig) -> PreinsertionPlan:
        return configured_preinsertion(args)

    def plan_candidate(
        self,
        candidate: GoalPlanCandidate,
        preinsertion: PreinsertionPlan,
        q_start: np.ndarray,
        T_eef_object: np.ndarray,
        T_object_eef: np.ndarray,
        candidate_count: int,
    ) -> dict | None:
        return _plan_goal_candidate(
            self.context,
            candidate,
            preinsertion,
            q_start,
            T_eef_object,
            T_object_eef,
            candidate_count,
        )


class TrajectoryBuilder:
    """Build, validate, and persist the controller-facing phase trajectory."""

    def __init__(self, context: PlanningContext):
        self.context = context

    def build(
        self,
        *,
        grasp: GraspPlan,
        selected_plan: dict,
        candidate: GoalPlanCandidate,
        preinsertion: PreinsertionPlan,
        T_eef_object: np.ndarray,
        source_xml: Path,
    ) -> dict:
        trajectory = _build_phase_waypoint_plan(
            ctx=self.context,
            grasp_plan=grasp,
            selected_plan=selected_plan,
            selected_candidate=candidate,
            pre=preinsertion,
            selected_T_eef_object=T_eef_object,
            source_xml=source_xml,
        )
        _validate_controller_trajectory(trajectory)
        return trajectory

    @staticmethod
    def write(path: Path, trajectory: dict) -> None:
        _write_trajectory(path, trajectory)


class Executor:
    """Execution boundary independent of OMPL, MuJoCo, and robot hardware.

    ``move_arm`` receives an ``(N, 7)`` path and the requested waypoint period.
    ``command_gripper`` receives ``"open"`` or ``"close"``. A real robot,
    simulator, or logger can implement those two callbacks without changing the
    planner.
    """

    def execute(
        self,
        trajectory: dict,
        *,
        move_arm: Callable[[np.ndarray, float], None],
        command_gripper: Callable[[str], None],
    ) -> None:
        _validate_controller_trajectory(trajectory)
        for phase in trajectory["phases"]:
            if phase["type"] == "move":
                move_arm(
                    np.asarray(phase["waypoints"], dtype=float),
                    float(phase.get("dt", trajectory.get("dt", 0.08))),
                )
            else:
                command_gripper(str(phase["command"]))


class Planner:
    """Compose the seven planning interfaces into one pick-and-place planner."""

    def __init__(
        self,
        args: PipelineConfig,
        xml_path: Path,
        *,
        grasp_observation: GraspObservation | None = None,
        anygrasp_config: AnyGraspConfig | None = None,
        grasp_choice_reader: Callable[[str], str] | None = None,
    ):
        self.args = args
        self.xml_path = Path(xml_path)
        self.grasp_observation = grasp_observation
        self.anygrasp_config = anygrasp_config
        self.grasp_choice_reader = grasp_choice_reader
        self.context = _make_context(args, self.xml_path)
        self.ik = IKSolver(self.context)
        self.collision = CollisionChecker(self.context)
        self.grasps = GraspApproachPlanner(self.context, self.collision)
        self.attached = AttachedObjectPlanner(self.context, self.ik, self.collision)
        self.insertion = InsertionPlanner(self.context, self.attached)
        self.trajectory_builder = TrajectoryBuilder(self.context)
        self.executor = Executor()

    def _candidate_batches(
        self,
        goal_pose: ObjectPose,
    ) -> Iterator[tuple[str, list[GoalFilteredGrasp]]]:
        """Bridge grasp generation to this planner's MuJoCo goal filter."""
        yield from grasp_candidate_batches(
            observation=self.grasp_observation,
            anygrasp_config=self.anygrasp_config,
            authored_configs=self.args.grasp_candidates,
            goal_filter=lambda candidates, max_results: self.grasps.filter_for_goal(
                candidates,
                goal_pose,
                max_results=max_results,
            ),
            choice_reader=self.grasp_choice_reader,
        )

    def plan(
        self,
        *,
        source_pose: ObjectPose,
        refined_goal_pose: ObjectPose,
        goal_passive_pose: ObjectPose | None,
        traj_dir: Path,
    ) -> PlanningResult | None:
        print("[planning] planning pick-and-place trajectory")
        pre = self.insertion.configure(self.args)
        failed = {
            "status": "failed",
            "reason": "OMPL planning failed for all equivalent goal Z rotations",
            "q_path": None,
        }
        selected: tuple[GraspPlan, GoalPlanCandidate, dict, np.ndarray, list] | None = None
        saw_grasp_approach = False
        saw_goal_feasible_grasp = False
        printed_attached_planning = False

        for _, candidate_batch in self._candidate_batches(refined_goal_pose):
            saw_goal_feasible_grasp = True
            for grasp in self.grasps.plan_approaches(source_pose, candidate_batch):
                saw_grasp_approach = True
                T_eef_object = self.grasps.attach(grasp)
                if T_eef_object is None:
                    failed["reason"] = "grasp attachment failed"
                    continue
                T_object_eef = np.linalg.inv(T_eef_object)

                retreat = self.attached.plan_to_configuration(
                    grasp.q_grasp,
                    grasp.q_pregrasp,
                    T_eef_object,
                    interpolate_count=GRASP_INTERPOLATE_COUNT,
                )
                if retreat is None:
                    failed["reason"] = "retreat from grasp to pregrasp failed"
                    continue

                candidates = self.attached.goal_candidates(
                    refined_goal_pose,
                    pre,
                    T_object_eef,
                    grasp.q_pregrasp,
                    grasp.goal_z_indices,
                    grasp.goal_q_by_index,
                )
                if not candidates:
                    target = "pre-insertion" if pre.settings["use_preinsertion"] else "goal"
                    failed["reason"] = f"{target} IK failed for all equivalent Z rotations"
                    continue

                if not printed_attached_planning:
                    print("[planning] planning attached motion")
                    printed_attached_planning = True

                for candidate in candidates:
                    plan = self.insertion.plan_candidate(
                        candidate,
                        pre,
                        grasp.q_pregrasp,
                        T_eef_object,
                        T_object_eef,
                        len(candidates),
                    )
                    if plan is None:
                        continue
                    plan["grasp_to_pregrasp_path"] = retreat
                    selected = grasp, candidate, plan, T_eef_object, candidates
                    break
                if selected is not None:
                    break
            if selected is not None:
                break

        trajectory_path = Path(traj_dir) / "trajectory_result.json"
        if not saw_goal_feasible_grasp:
            print("[planning:grasp] no grasp is collision-free at the goal")
            return None
        if not saw_grasp_approach:
            print("[planning:grasp] no collision-free grasp approach")
            return None
        if selected is None:
            self.trajectory_builder.write(trajectory_path, failed)
            return PlanningResult(source_pose, refined_goal_pose, pre.settings, failed)

        grasp, candidate, selected_plan, T_eef_object, candidates = selected
        refined_goal_pose = ObjectPose(
            refined_goal_pose.name,
            candidate.goal_T,
            refined_goal_pose.confidence,
            f"{refined_goal_pose.note}; equivalent Z rotation sample {candidate.index}",
        )
        _log_selected_poses(
            candidate.goal_T,
            goal_passive_pose,
            grasp_T=grasp.source_pose.T_world_object,
            preinsertion_T=selected_plan["preinsertion_T"],
        )
        trajectory = self.trajectory_builder.build(
            grasp=grasp,
            selected_plan=selected_plan,
            candidate=candidate,
            preinsertion=pre,
            T_eef_object=T_eef_object,
            source_xml=self.xml_path,
        )
        if len(candidates) > 1:
            print(f"[planning:select] selected goal Z sample {candidate.index}")
        self.trajectory_builder.write(trajectory_path, trajectory)
        return PlanningResult(grasp.source_pose, refined_goal_pose, pre.settings, trajectory)


def plan_pick_place(
    *,
    args: PipelineConfig,
    xml_path: Path,
    source_pose: ObjectPose,
    refined_goal_pose: ObjectPose,
    goal_passive_pose: ObjectPose | None,
    traj_dir: Path,
    grasp_observation: GraspObservation | None = None,
    anygrasp_config: AnyGraspConfig | None = None,
    grasp_choice_reader: Callable[[str], str] | None = None,
    await_next_step=None,
) -> PlanningResult | None:
    """Compatibility facade used by the pipeline orchestrator."""
    del await_next_step
    return Planner(
        args,
        xml_path,
        grasp_observation=grasp_observation,
        anygrasp_config=anygrasp_config,
        grasp_choice_reader=grasp_choice_reader,
    ).plan(
        source_pose=source_pose,
        refined_goal_pose=refined_goal_pose,
        goal_passive_pose=goal_passive_pose,
        traj_dir=traj_dir,
    )
