"""Run from the repository root: python scripts/view_ensemble.py lid_on_teapot --stage 0 [--video].
Enter: settle/next; C: select a passing settled world; T: translate/rotate;
arrows/PageUp/PageDown: teleoperate; [ / ]: select cell; R: replay;
F: focus; G: grid; mouse drag: orbit; wheel: zoom; Esc: quit.
Rebuilds an artifact-local ensemble XML from cached artifacts. Does not rebuild
meshes or query a VLM.
"""
import argparse
import copy
import cv2
import json
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class Preview:
    def __init__(self, xml, vlm, dimensions, candidate_dir=None, meshes=None,
                 render_config=None, seed=0):
        print(f"Loading {xml} ...", flush=True)
        start = time.perf_counter()
        self.m = m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ids = sorted({int(match[1]) for b in range(m.nbody)
                      if (match := re.match(r'cell(\d+)_active_', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ''))})
        if not ids: raise ValueError('XML has no cellN_active_ bodies; use the generated ensemble XML.')
        def body(cell, role):
            return next(b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith(f'cell{cell}_{role}_'))
        self.active = np.array([body(i, 'active') for i in ids])
        self.passive = np.array([body(i, 'passive') for i in ids])
        self.tables = np.array([m.geom(f'cell{i}_table').id for i in ids])
        self.qadr = m.jnt_qposadr[m.body_jntadr[self.active]]
        self.initial = d.qpos[self.qadr[:, None] + np.arange(7)].copy()
        self.centers = d.geom_xpos[self.tables, :2].copy()
        self.table_half_size = m.geom_size[self.tables, :2].copy()
        self.rpassive = d.xmat[self.passive[0]].reshape(3, 3).copy()
        self.owner = np.full(m.ngeom, -1, int)
        self.role = np.full(m.ngeom, -1, int)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ''
                match = re.match(r'cell(\d+)_(active|passive|obstacle)_', name)
                if match:
                    self.owner[g] = ids.index(int(match[1]))
                    self.role[g] = {'active': 0, 'passive': 1, 'obstacle': 3}[match[2]]
                    break
                b = int(m.body_parentid[b])
        self.owner[self.tables], self.role[self.tables] = np.arange(len(ids)), 2
        self.base_rgba = m.geom_rgba.copy()
        self.n, self.count, self.current, self.valid = len(ids), 0, None, None
        self.candidate_dir, self.meshes = candidate_dir, meshes
        self.render_config, self.rng = render_config, np.random.default_rng(seed)
        self.v = json.loads((vlm / 'validation_program.json').read_text())
        jt = json.loads((vlm / 'translation_constraints.json').read_text())['translation']
        jr = json.loads((vlm / 'rotation_constraints.json').read_text())['rotation']
        length = max(dimensions)
        self.translation = np.array([{'correct': .25, 'normal_search': .75, 'major_revision': 1.5}[jt[k]['assessment']] * length for k in 'xyz'])
        self.bias = [jt[k]['direction'] for k in 'xyz']
        self.rotation = np.array([{'correct': 15, 'normal_search': 60, 'major_revision': 120}[jr[k]['assessment']] for k in ('roll','pitch','yaw')])
        self.geometric = self.v['validation_mode'] == 'geometric_only'
        if not self.geometric:
            self.plimit = np.array([{'strict': .25, 'moderate': .75, 'permissive': 1.5}[self.v['position_drift_tolerance'][k]] * length for k in 'xyz'])
            self.rlimit = np.array([{'strict': 7.5, 'moderate': 30, 'permissive': 60}[self.v['orientation_drift_tolerance'][k]] for k in ('roll','pitch','yaw')])
        print(f"Loaded {self.n} cells, {m.ngeom} geoms in {time.perf_counter()-start:.1f}s. Mode: {self.v['validation_mode']}", flush=True)

    def contacts(self):
        pen, touch = np.zeros(self.n), np.zeros((self.n, 4), bool)
        for c in self.d.contact:
            for a, b in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if self.role[a] != 0: continue
                i = self.owner[a]
                pen[i] = max(pen[i], -float(c.dist))
                role = self.role[b] if self.owner[b] == i else 3
                if role >= 0: touch[i, role] = True
        return pen, touch

    def begin(self, replay=False):
        q = self.initial.copy()
        if replay and self.current is not None:
            q = self.current.copy()
        elif self.count:
            shift = np.array([np.random.uniform(0, r) if b == 'positive' else np.random.uniform(-r, 0) if b == 'negative' else np.random.uniform(-r, r) for r, b in zip(self.translation, self.bias)])
            q[:, :3] += shift
            delta = Rotation.from_euler('xyz', np.random.uniform(-self.rotation, self.rotation), degrees=True)
            rotation = Rotation.from_quat(q[:, [4,5,6,3]]) * delta
            q[:, 3:] = rotation.as_quat()[:, [3,0,1,2]]
        if not replay: self.count += 1
        self.current = q.copy()
        mujoco.mj_resetData(self.m, self.d)
        self.m.geom_rgba[:] = self.base_rgba
        self.d.qpos[self.qadr[:, None] + np.arange(7)] = q
        mujoco.mj_forward(self.m, self.d)
        self.valid = None
        print(f"Sample {self.count}{' (replay)' if replay else ''}: ready for teleoperation", flush=True)

    def start(self):
        # Same thresholds as the current validation_program_from_json / EnsembleTwin.
        self.valid_initial = self.contacts()[0] <= (.0008 if self.geometric else .05)
        self.current = self.d.qpos[self.qadr[:, None] + np.arange(7)].copy()
        print(f"Sample {self.count}: settling", flush=True)

    def finish(self):
        mujoco.mj_forward(self.m, self.d)
        pen, touch = self.contacts()
        failures = {
            'initial_pen': ~self.valid_initial,
            'final_pen': pen > .0008,
            'obstacle': touch[:, 3],
        }
        for name, col in (('active_passive', 1), ('active_table', 2)):
            if name in self.v.get('required_contacts', []): failures[f'missing_{name}'] = ~touch[:, col]
            if name in self.v.get('forbidden_contacts', []): failures[f'forbidden_{name}'] = touch[:, col]
        failures['off_table'] = ~np.all(np.abs(self.d.xpos[self.active, :2] - self.centers) <= self.table_half_size, axis=1)
        if not self.geometric:
            failures['position_drift'] = ~np.all(np.abs(self.d.xpos[self.active] - self.current[:, :3]) <= self.plimit, axis=1)
            initial_r = Rotation.from_quat(self.current[:, [4,5,6,3]]).as_matrix()
            delta = np.swapaxes(initial_r, 1, 2) @ self.d.xmat[self.active].reshape(-1,3,3)
            failures['rotation_drift'] = ~np.all(np.abs(Rotation.from_matrix(delta).as_euler('xyz', degrees=True)) <= self.rlimit, axis=1)
            if any(w.number for w in self.d.warning) or not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all(): failures['unstable'] = np.ones(self.n, bool)
        relative = self.d.xpos[self.active] - self.d.xpos[self.passive]
        for relation in self.v.get('xyz_relations', []):
            value = relative[:, 'xyz'.index(relation['axis'])]
            failures[f"{relation['axis']}_{relation['relation']}"] = value <= 0 if relation['relation'] == 'greater_than' else value >= 0
        valid = ~np.logical_or.reduce(list(failures.values()))
        for g in np.flatnonzero(self.role == 0):
            self.m.geom_rgba[g] = (0.2, .85, .3, 1) if valid[self.owner[g]] else (.9, .2, .2, 1)
        tally = ', '.join(f'{name}={mask.sum()}' for name, mask in failures.items() if mask.any()) or 'none'
        print(f"PASS {valid.sum()}/{self.n} | failures: {tally}", flush=True)
        self.valid = valid
        return valid

    def teleoperate(self, axis, amount, rotate=False):
        q = self.d.qpos[self.qadr[:, None] + np.arange(7)].copy()
        if rotate:
            delta = np.zeros(3); delta[axis] = amount
            r = Rotation.from_quat(q[:, [4,5,6,3]]) * Rotation.from_rotvec(delta)
            q[:, 3:] = r.as_quat()[:, [3,0,1,2]]
        else:
            q[:, axis] += amount
        self.d.qpos[self.qadr[:, None] + np.arange(7)] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def body_pose(self, body):
        T = np.eye(4); T[:3, 3] = self.d.xpos[body]
        T[:3, :3] = self.d.xmat[body].reshape(3, 3)
        return T

    def select_candidate(self):
        passing = np.flatnonzero(self.valid) if self.valid is not None else []
        if not len(passing):
            print('No passing settled world is available.', flush=True); return
        world = int(self.rng.choice(passing))
        Ta, Tp = self.body_pose(self.active[world]), self.body_pose(self.passive[world])
        Tpa = np.linalg.inv(Tp) @ Ta
        xyz = Tpa[:3, 3]; rpy = Rotation.from_matrix(Tpa[:3, :3]).as_euler('xyz', degrees=True)
        print(f'Selected world {world}: passive_T_active xyz={xyz.round(6).tolist()} '
              f'rpy_deg={rpy.round(3).tolist()}\n{np.array2string(Tpa, precision=6)}', flush=True)
        if self.candidate_dir is None: return
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        stem = self.candidate_dir/f'interactive_candidate_{self.count:03d}_world_{world:02d}'
        np.savez_compressed(stem.with_suffix('.npz'), world=world, T_passive_active=Tpa,
                           T_active_cell=Ta, T_passive_cell=Tp)
        from src.render import render_for_vlm
        render_for_vlm(active_path=self.meshes[0], passive_path=self.meshes[1],
                       output_path=stem.with_suffix('.png'), T_passive_active=Tpa,
                       rendered_frame='active', **self.render_config)
        print(f'Rendered {stem.with_suffix(".png")}', flush=True)


class VideoRecorder:
    def __init__(self, path, fps=30, width=1280, height=720):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.size, self.period, self.next = (width, height), 1/fps, 0.0
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, self.size)
        if not self.writer.isOpened(): raise RuntimeError(f'Could not open video writer: {path}')
        self.path = path

    def capture(self, m, d, option, camera, scene, context):
        now = time.perf_counter()
        if now < self.next: return
        self.next = now + self.period
        w, h = self.size; viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(m, d, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.empty((h, w, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        self.writer.write(cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release(); print(f'Video saved to {self.path}', flush=True)


def show(p, steps, speed, video_path=None, fps=30, translation_step=.005, rotation_step=5.):
    import glfw
    if not glfw.init(): raise RuntimeError('GLFW could not open a display.')
    window = glfw.create_window(1400, 900, 'Ensemble: Enter settle/next | C candidate | T translate/rotate', None, None)
    if not window:
        glfw.terminate(); raise RuntimeError('Could not create viewer window.')
    glfw.make_context_current(window); glfw.swap_interval(1)
    scene = mujoco.MjvScene(p.m, maxgeom=p.m.ngeom + 1000)
    context = mujoco.MjrContext(p.m, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera, record_camera, option = mujoco.MjvCamera(), mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(camera); mujoco.mjv_defaultCamera(record_camera)
    recorder = VideoRecorder(video_path, fps) if video_path else None
    command, selected, remaining, phase, rotate = [], 0, 0, 'ready', False
    def grid(target=camera, aspect=1400/900):
        span = np.ptp(p.centers, axis=0) + 2*p.table_half_size.max(axis=0)
        target.lookat[:] = [*p.centers.mean(axis=0), p.d.geom_xpos[p.tables[0], 2]]
        target.distance = 1.08*max(span[0]/aspect, span[1])/(2*np.tan(np.deg2rad(22.5)))
        target.azimuth, target.elevation = 90, -78
    def focus():
        camera.lookat[:] = p.d.xpos[p.passive[selected]]
        camera.distance, camera.elevation = .65, -30
    def key(w, k, sc, action, mods):
        nonlocal selected, rotate
        if action not in (glfw.PRESS, glfw.REPEAT): return
        if k == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
        elif action == glfw.PRESS and k in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and not remaining:
            command.append('evaluate' if phase == 'ready' else 'next')
        elif action == glfw.PRESS and k == glfw.KEY_R and not remaining: command.append('replay')
        elif action == glfw.PRESS and k == glfw.KEY_C and phase == 'evaluated': command.append('candidate')
        elif action == glfw.PRESS and k == glfw.KEY_T and phase == 'ready':
            rotate = not rotate; print(f"Teleoperation mode: {'rotation' if rotate else 'translation'}", flush=True)
        elif k == glfw.KEY_G: grid()
        elif k == glfw.KEY_F: focus()
        elif k in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
            selected = (selected + (1 if k == glfw.KEY_RIGHT_BRACKET else -1)) % p.n
            focus(); print(f'Viewing cell {selected}', flush=True)
        elif phase == 'ready' and k in (glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_DOWN,
                                        glfw.KEY_UP, glfw.KEY_PAGE_DOWN, glfw.KEY_PAGE_UP):
            sign = 1 if k in (glfw.KEY_RIGHT, glfw.KEY_UP, glfw.KEY_PAGE_UP) else -1
            if rotate:
                axis = 2 if k in (glfw.KEY_LEFT, glfw.KEY_RIGHT) else 1 if k in (glfw.KEY_DOWN, glfw.KEY_UP) else 0
                p.teleoperate(axis, sign*np.deg2rad(rotation_step), True)
            else:
                axis = 0 if k in (glfw.KEY_LEFT, glfw.KEY_RIGHT) else 1 if k in (glfw.KEY_DOWN, glfw.KEY_UP) else 2
                p.teleoperate(axis, sign*translation_step)
    last = [0., 0.]
    def mouse(w, x, y):
        dx, dy = x-last[0], y-last[1]; last[:] = [x,y]
        if glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            camera.azimuth -= .3*dx; camera.elevation = np.clip(camera.elevation-.3*dy, -89, 89)
    glfw.set_key_callback(window, key)
    glfw.set_cursor_pos_callback(window, mouse)
    glfw.set_scroll_callback(window, lambda w,x,y: setattr(camera, 'distance', max(.05, camera.distance*np.exp(-.1*y))))
    grid(); grid(record_camera, 1280/720); p.begin()
    print('Ready. Arrows/PageUp/PageDown: move; T: translate/rotate; Enter: settle.', flush=True)
    try:
        while not glfw.window_should_close(window):
            frame = time.perf_counter()
            if command:
                action = command.pop(0)
                if action in ('next', 'replay'):
                    p.begin(action == 'replay'); phase = 'ready'
                elif action == 'candidate':
                    p.select_candidate(); glfw.make_context_current(window)
                elif action == 'evaluate':
                    p.start(); remaining = 0 if p.geometric else steps
                    if p.geometric: p.finish(); phase = 'evaluated'
            if remaining:
                chunk = min(remaining, max(1, round(speed / (60*p.m.opt.timestep))))
                mujoco.mj_step(p.m, p.d, nstep=chunk)
                remaining -= chunk
                mujoco.mj_forward(p.m, p.d)
                if not remaining: p.finish(); phase = 'evaluated'
            if recorder: recorder.capture(p.m, p.d, option, record_camera, scene, context)
            width, height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(p.m, p.d, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(mujoco.MjrRect(0,0,width,height), scene, context)
            glfw.swap_buffers(window); glfw.poll_events()
            time.sleep(max(0, 1/60 - (time.perf_counter()-frame)))
    finally:
        if recorder: recorder.close()
        context.free(); scene.free(); glfw.destroy_window(window); glfw.terminate()


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Required artifact is missing: {path}')
    return path


def rebuild_ensemble(root, benchmark, stage):
    """Recreate the viewer XML entirely from current cached artifacts."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.config import BENCHMARK_CONFIGS, OBJECT_CATALOG, T_mujoco_board
    from src.digital_twin import EnsembleTwin, SceneBuilder
    from src.vlm import load_sampling_constraints, load_validation_program

    config = BENCHMARK_CONFIGS[benchmark]
    try:
        active = config['active'][stage]
        passive = config['passive'][stage]
    except IndexError as exc:
        raise ValueError(f'{benchmark} has no stage {stage}') from exc

    stage_dir = root/'artifacts'/f'{benchmark}_stage{stage}'
    artifact = stage_dir/'estimated'
    perception = artifact/'perception'
    vlm = artifact/'vlm'
    dtwin = artifact/'dtwin'
    estimate_root = root/'artifacts'/'object_estimation'

    catalog = copy.deepcopy(OBJECT_CATALOG)
    estimates = {}

    def load_estimate(name):
        if name not in estimates:
            path = require(estimate_root/name/'estimate.json')
            estimate = json.loads(path.read_text())

            # estimate.json may have been produced on another computer and its
            # paths may therefore contain an obsolete absolute repository root.
            # Reconstruct them from the portable cache layout instead of using
            # the stored parent directories.
            estimate['mesh'] = str(require(
                estimate_root/name/Path(estimate['mesh']).name))
            mjcf = estimate['mjcf']
            if mjcf['type'] == 'mesh':
                remeshed = estimate_root/f'{name}_remeshed'
                mjcf['asset_xml'] = str(require(
                    remeshed/Path(mjcf['asset_xml']).name))
                mjcf['body_xml'] = str(require(
                    remeshed/Path(mjcf['body_xml']).name))

            estimates[name] = estimate
            catalog[name].update(estimates[name])
        return estimates[name]

    active_estimate = load_estimate(active)
    passive_estimate = load_estimate(passive)

    # Reproduce the transforms used by main.py from saved perception outputs.
    T_cam_board = np.load(require(root/'artifacts'/'calibration'/'extrinsics.npy'))
    T_cam_passive_source = np.load(require(perception/'source_passive_pose_cam.npy'))
    T_cam_active_edited = np.load(require(perception/'edited_active_pose_cam.npy'))
    T_cam_passive_edited = np.load(require(perception/'edited_passive_pose_cam.npy'))
    T_passive = T_mujoco_board @ np.linalg.inv(T_cam_board) @ T_cam_passive_source
    T_goal = T_passive @ np.linalg.inv(T_cam_passive_edited) @ T_cam_active_edited

    payload = json.loads(require(dtwin/'worlds.json').read_text())
    hypotheses = payload['hypotheses']
    saved_worlds = payload['worlds']
    if not hypotheses or not saved_worlds:
        raise ValueError(f'No ensemble definitions in {dtwin/"worlds.json"}')

    # EnsembleTwin uses one ordered scale set for both objects. Preserve the
    # exact order recorded when the artifact was created and verify the result.
    scales = tuple(dict.fromkeys(world['active_scale'] for world in saved_worlds))
    if set(scales) != {world['passive_scale'] for world in saved_worlds}:
        raise ValueError('Saved active/passive scale sets differ; cannot rebuild this ensemble')

    obstacles = []
    for previous_stage in range(stage):
        name = config['active'][previous_stage]
        if name in (active, passive) or any(name == old for old, _ in obstacles):
            continue
        load_estimate(name)
        previous = root/'artifacts'/f'{benchmark}_stage{previous_stage}'/'estimated'/'dtwin'
        pose = np.load(require(previous/'T_mujoco_active_goal_selected.npy'))
        obstacles.append((name, pose))

    dimensions = active_estimate['dimensions']
    S = load_sampling_constraints(
        require(vlm/'translation_constraints.json'),
        require(vlm/'rotation_constraints.json'), dimensions)
    V = load_validation_program(require(vlm/'validation_program.json'), dimensions)

    xml = dtwin/'view_ensemble.xml'
    xml.parent.mkdir(parents=True, exist_ok=True)
    ensemble = EnsembleTwin(SceneBuilder(catalog), active, passive, T_passive,
                            T_goal, obstacles, hypotheses, V, S, xml, scales=scales)
    if ensemble.specs != saved_worlds:
        raise ValueError('Rebuilt world ordering differs from saved worlds.json')
    del ensemble
    return xml, artifact, active_estimate, passive_estimate, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark')
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--video', action='store_true', help='Record the session to dtwin/sampling.mp4')
    args = parser.parse_args()
    if args.stage < 0:
        parser.error('stage must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    xml, artifact, active_estimate, passive_estimate, config = rebuild_ensemble(
        root, args.benchmark, args.stage)
    vlm = artifact/'vlm'
    p = Preview(xml, vlm, active_estimate['dimensions'],
                artifact/'dtwin'/'interactive_candidates',
                [active_estimate['mesh'], passive_estimate['mesh']],
                config['render_config'][1])
    video_path = artifact/'dtwin'/'sampling.mp4' if args.video else None
    show(p, steps=500, speed=1., video_path=video_path)



if __name__ == '__main__': main()
