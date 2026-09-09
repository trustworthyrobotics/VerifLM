"""Run from the repository root: python view_ensemble.py lid_on_teapot.
Enter: evaluate (original first, new samples afterward); R: replay; F: focus;
G: grid; Left/Right: select cell; mouse drag: orbit; wheel: zoom; Esc: quit.
Only loads existing XML/JSON. Does not rebuild meshes or query a VLM.
"""
import argparse
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class Preview:
    def __init__(self, xml, vlm, dimensions):
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
        self.n, self.count, self.current = len(ids), 0, None
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
        # Same thresholds as the current validation_program_from_json / EnsembleTwin.
        self.valid_initial = self.contacts()[0] <= (.0008 if self.geometric else .05)
        print(f"Evaluation {self.count}{' (replay)' if replay else ''}: starting", flush=True)

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
        return valid


def show(p, steps, speed):
    import glfw
    if not glfw.init(): raise RuntimeError('GLFW could not open a display.')
    window = glfw.create_window(1400, 900, 'Ensemble sampling: Enter evaluate | R replay | F focus | G grid', None, None)
    if not window:
        glfw.terminate(); raise RuntimeError('Could not create viewer window.')
    glfw.make_context_current(window); glfw.swap_interval(1)
    scene = mujoco.MjvScene(p.m, maxgeom=p.m.ngeom + 1000)
    context = mujoco.MjrContext(p.m, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    command, selected, remaining = [], 0, 0
    def grid():
        camera.lookat[:] = [*p.centers.mean(axis=0), p.d.geom_xpos[p.tables[0], 2]]
        camera.distance = max(1., np.ptp(p.centers, axis=0).max() * 1.5)
        camera.azimuth, camera.elevation = 90, -65
    def focus():
        camera.lookat[:] = p.d.xpos[p.passive[selected]]
        camera.distance, camera.elevation = .65, -30
    def key(w, k, sc, action, mods):
        nonlocal selected
        if action != glfw.PRESS: return
        if k == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
        elif k in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER, glfw.KEY_R) and not remaining:
            command[:] = [k == glfw.KEY_R]
        elif k == glfw.KEY_G: grid()
        elif k == glfw.KEY_F: focus()
        elif k in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            selected = (selected + (1 if k == glfw.KEY_RIGHT else -1)) % p.n
            focus(); print(f'Viewing cell {selected}', flush=True)
    last = [0., 0.]
    def mouse(w, x, y):
        dx, dy = x-last[0], y-last[1]; last[:] = [x,y]
        if glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            camera.azimuth -= .3*dx; camera.elevation = np.clip(camera.elevation-.3*dy, -89, 89)
    glfw.set_key_callback(window, key)
    glfw.set_cursor_pos_callback(window, mouse)
    glfw.set_scroll_callback(window, lambda w,x,y: setattr(camera, 'distance', max(.05, camera.distance*np.exp(-.1*y))))
    grid(); print('Paused. Press Enter in the viewer to evaluate. F: close-up; arrows: change cell; G: overview.', flush=True)
    try:
        while not glfw.window_should_close(window):
            frame = time.perf_counter()
            if command:
                p.begin(command.pop()); remaining = 0 if p.geometric else steps
                if p.geometric: p.finish()
            if remaining:
                chunk = min(remaining, max(1, round(speed / (60*p.m.opt.timestep))))
                mujoco.mj_step(p.m, p.d, nstep=chunk)
                remaining -= chunk
                mujoco.mj_forward(p.m, p.d)
                if not remaining: p.finish()
            width, height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(p.m, p.d, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(mujoco.MjrRect(0,0,width,height), scene, context)
            glfw.swap_buffers(window); glfw.poll_events()
            time.sleep(max(0, 1/60 - (time.perf_counter()-frame)))
    finally:
        context.free(); scene.free(); glfw.destroy_window(window); glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark')
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root (default: current directory)')
    parser.add_argument('--xml', type=Path, help='Override ensemble XML path')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--speed', type=float, default=1., help='Playback speed, e.g. 0.25 for slow motion')
    parser.add_argument('--headless', action='store_true', help='Evaluate original proposal once without a viewer')
    args = parser.parse_args()
    if args.steps < 1 or args.speed <= 0: parser.error('steps and speed must be positive')
    root = args.root.resolve()
    xml = args.xml or root/'models'/'generated_xmls'/f'{args.benchmark}_stage{args.stage}_estimated_ensemble.xml'
    vlm = root/'artifacts'/f'{args.benchmark}_stage{args.stage}'/'estimated'/'vlm'
    # Cell 0's active body name identifies the shared estimated-object cache.
    import xml.etree.ElementTree as ET
    body = next(b for b in ET.parse(xml).getroot().iter('body') if b.get('name','').startswith('cell0_active_'))
    name = body.get('name').removeprefix('cell0_active_')
    dimensions = json.loads((root/'artifacts'/'object_estimation'/name/'estimate.json').read_text())['dimensions']
    p = Preview(xml, vlm, dimensions)
    if args.headless:
        p.begin()
        if not p.geometric: mujoco.mj_step(p.m, p.d, nstep=args.steps)
        p.finish()
    else: show(p, args.steps, args.speed)


if __name__ == '__main__': main()
