class DigitalTwin:
    AXIS = {"x": 0, "y": 1, "z": 2}

    def __init__(self, scene: Scene, active_dimensions, sim_steps=500):
        self.scene, self.model = scene, scene.model
        self.active_dimensions = np.asarray(active_dimensions, dtype=float)
        self.sim_steps = sim_steps

        joint_id = int(self.model.body_jntadr[scene.active_body_id])
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("[digital-twin]: active body must have a freejoint")
        self.qadr, self.dadr = int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

        self.active_body_ids = self._subtree_ids(scene.active_body_id)
        self.passive_body_ids = self._subtree_ids(scene.passive_body_id)

        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(self.model, self.state_spec))
        mujoco.mj_getState(self.model, scene.data, self.state, self.state_spec)

        self.T_goal = self._mocap_pose(scene.goal_mocap_id)

    def sample_and_validate(self, S: SamplingConstraints, V: ValidationProgram, n_samples=500, max_valid=10,
                            visualize=False, pause_on_valid=False):
        if not visualize:
            return self._sample(S, V, n_samples, max_valid)

        import time
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.scene.data) as viewer:
            def show(T):
                self.show_goal(T)
                viewer.sync()
                time.sleep(0.005)

            return self._sample(S, V, n_samples, max_valid, show, pause_on_valid)

    def _sample(self, S, V, n_samples, max_valid, show=None, pause_on_valid=False):
        candidates = []

        for T in [self.T_goal] + [S.sample(self.T_goal) for _ in range(n_samples)]:
            if show: show(T)

            valid, T_final, reason = self.validate(T, V)
            if not valid: continue

            candidates.append(T_final)
            print(f"[digital-twin]: valid candidate {len(candidates)}")
            if show: show(T_final)
            if pause_on_valid: input("[digital-twin]: press Enter to continue")
            if len(candidates) >= max_valid: break

        return candidates

    def validate(self, T_sample, V: ValidationProgram):
        data = self._fresh_data()
        self._set_active_pose(data, T_sample)

        # Evaluate proposed pose.
        try: mujoco.mj_forward(self.model, data)
        except mujoco.FatalError as e: return False, T_sample, f"mj_forward failed: {e}"

        initial_pen, contacts = self._contact_state(data)

        # Geometric-only validation skips physics.
        if V.validation_mode == "geometric_only":
            return self._check_final(data, T_sample, V, check_drift=False)

        # Reject invalid initial configurations.
        if initial_pen > V.max_initial_penetration:
            return False, T_sample, f"initial penetration {initial_pen:.6f}"
        forbidden = set(V.forbidden_contacts) & contacts
        if forbidden:
            return False, T_sample, f"initial forbidden contacts: {sorted(forbidden)}"

        # Simulate under gravity for a fixed duration.
        for _ in range(self.sim_steps):
            try: mujoco.mj_step(self.model, data)
            except mujoco.FatalError as e: return False, T_sample, f"mj_step failed: {e}"

            _, contacts = self._contact_state(data)
            forbidden = set(V.forbidden_contacts) & contacts
            if forbidden:
                return False, T_sample, f"forbidden contacts while settling: {sorted(forbidden)}"

        # Validate the resulting settled pose.
        mujoco.mj_forward(self.model, data)
        return self._check_final(data, T_sample, V, check_drift=True)

    def _check_final(self, data, T_sample, V, check_drift):
        penetration, contacts = self._contact_state(data)
        if penetration > V.max_final_penetration:
            return False, T_sample, f"final penetration {penetration:.6f}"

        missing = set(V.required_contacts) - contacts
        forbidden = set(V.forbidden_contacts) & contacts
        if missing: return False, T_sample, f"missing required contacts: {sorted(missing)}"
        if forbidden: return False, T_sample, f"forbidden contacts: {sorted(forbidden)}"
        if not self._relations_valid(data, V): return False, T_sample, "XYZ relation failed"

        T_final = self._body_pose(data, self.scene.active_body_id)
        if check_drift and not self._drift_valid(T_sample, T_final, V):
            return False, T_final, "pose drift exceeded tolerance"
        return True, T_final, "valid"

    def _drift_valid(self, T0, T1, V):
        position_drift = np.abs(T1[:3, 3] - T0[:3, 3])
        position_limit = max(self.active_dimensions) * np.array(
            [V.x_drift_tolerance, V.y_drift_tolerance, V.z_drift_tolerance]
        )
        if np.any(position_drift > position_limit): return False

        R_delta = T0[:3, :3].T @ T1[:3, :3]
        rpy_drift = np.abs(Rotation.from_matrix(R_delta).as_euler("xyz", degrees=True))
        rpy_limit = np.array([V.roll_drift_tolerance, V.pitch_drift_tolerance, V.yaw_drift_tolerance])
        return not np.any(rpy_drift > rpy_limit)

    def _relations_valid(self, data, V):
        active, passive = data.xpos[self.scene.active_body_id], data.xpos[self.scene.passive_body_id]

        for r in V.xyz_relations:
            a, p = active[self.AXIS[r.axis]], passive[self.AXIS[r.axis]]
            if r.relation == "greater_than" and a <= p: return False
            if r.relation == "less_than" and a >= p: return False

        return True

    def _contact_state(self, data):
        max_pen, contacts = 0.0, set()

        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            a1, a2 = b1 in self.active_body_ids, b2 in self.active_body_ids
            if not (a1 or a2): continue

            max_pen = max(max_pen, max(0.0, -float(c.dist)))
            if (a1 and b2 in self.passive_body_ids) or (a2 and b1 in self.passive_body_ids):
                contacts.add("active_passive")
            if (a1 and g2 == self.scene.table_geom_id) or (a2 and g1 == self.scene.table_geom_id):
                contacts.add("active_table")

        return max_pen, contacts

    def _fresh_data(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, data, self.state, self.state_spec)
        return data

    def _set_active_pose(self, data, T):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        data.qpos[self.qadr:self.qadr + 7] = [*T[:3, 3], q[3], q[0], q[1], q[2]]
        data.qvel[self.dadr:self.dadr + 6] = 0

    def show_goal(self, T):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.scene.data.mocap_pos[self.scene.goal_mocap_id] = T[:3, 3]
        self.scene.data.mocap_quat[self.scene.goal_mocap_id] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(self.model, self.scene.data)

    def _subtree_ids(self, root):
        ids = set()
        for body in range(self.model.nbody):
            parent = body
            while parent:
                if parent == root:
                    ids.add(body)
                    break
                parent = int(self.model.body_parentid[parent])
        return frozenset(ids)

    def _body_pose(self, data, body_id):
        T = np.eye(4)
        T[:3, 3], T[:3, :3] = data.xpos[body_id], data.xmat[body_id].reshape(3, 3)
        return T

    def _mocap_pose(self, mocap_id):
        T = np.eye(4)
        q = self.scene.data.mocap_quat[mocap_id]
        T[:3, 3] = self.scene.data.mocap_pos[mocap_id]
        T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return T