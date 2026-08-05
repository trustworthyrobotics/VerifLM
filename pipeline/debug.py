from __future__ import annotations

import select
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from pipeline.core import ObjectMask, StagePaths


_WIN: Optional[str] = None


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _same_h(*imgs: np.ndarray) -> list[np.ndarray]:
    h = max(img.shape[0] for img in imgs)
    out = []
    for img in imgs:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h))
        out.append(img)
    return out


def _show(title: str, img: np.ndarray) -> None:
    global _WIN
    if _WIN and _WIN != title:
        cv2.destroyWindow(_WIN)
    _WIN = title
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    for _ in range(10):
        cv2.waitKey(50)


def _close() -> None:
    global _WIN
    if _WIN:
        cv2.destroyWindow(_WIN)
        cv2.waitKey(1)
        _WIN = None


def wait_for_enter(prompt: str, *, pump_cv2: bool = False) -> None:
    if not sys.stdin.isatty():
        print(f"[debug] non-interactive stdin; cannot wait for Enter: {prompt}")
        if pump_cv2:
            _close()
        return

    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)

    if not pump_cv2:
        try:
            input(f"\n{prompt}")
        except (KeyboardInterrupt, EOFError):
            pass
        return

    done = threading.Event()

    def _read() -> None:
        try:
            input(f"\n{prompt}")
        except (KeyboardInterrupt, EOFError):
            pass
        done.set()

    threading.Thread(target=_read, daemon=True).start()
    old = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        while not done.wait(timeout=0.05):
            cv2.waitKey(1)
    finally:
        signal.signal(signal.SIGINT, old)
        _close()


def _show_and_wait(title: str, img: np.ndarray, prompt: str = "Press Enter to continue...") -> None:
    _show(title, img)
    wait_for_enter(prompt, pump_cv2=True)


def _full_image_mask(name: str, image_bgr: np.ndarray) -> ObjectMask:
    h, w = image_bgr.shape[:2]
    return ObjectMask(
        name=name,
        mask=np.full((h, w), 255, dtype=np.uint8),
        bbox_xyxy=(0, 0, w, h),
        score=0.0,
    )


def _mask_pointcloud_preview_bgr(
    rgb_bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    K: np.ndarray,
    title: str,
    size: tuple[int, int] = (640, 360),
) -> np.ndarray:
    w, h = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if depth_m is None or mask is None:
        cv2.putText(canvas, "missing depth or mask", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        return canvas

    if rgb_bgr.shape[:2] != depth_m.shape[:2]:
        rgb_bgr = cv2.resize(rgb_bgr, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != depth_m.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        cv2.putText(canvas, "empty mask", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        return canvas

    z_all = depth_m[ys, xs].astype(np.float64)
    plausible = np.isfinite(z_all) & (z_all > 0.05) & (z_all < 5.0)
    sentinel = np.isclose(z_all, 65.535, atol=0.001)
    valid_frac = float(plausible.mean())
    sentinel_frac = float(sentinel.mean())
    if plausible.sum() < 4:
        cv2.putText(canvas, f"no plausible depth points ({valid_frac:.1%} valid)", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(canvas, f"65.535m sentinel: {sentinel_frac:.1%}", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        return canvas

    xs = xs[plausible]
    ys = ys[plausible]
    z = z_all[plausible]
    colors = rgb_bgr[ys, xs]
    if len(xs) > 40000:
        idx = np.linspace(0, len(xs) - 1, 40000).astype(int)
        xs, ys, z, colors = xs[idx], ys[idx], z[idx], colors[idx]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy

    cv2.putText(canvas, f"valid={valid_frac:.1%}  z_med={np.median(z):.3f}m  65.535={sentinel_frac:.1%}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 230, 255), 1)

    views = [
        ("XY", x, y, (0, 84, w // 3, h - 1)),
        ("XZ", x, z, (w // 3, 84, 2 * w // 3, h - 1)),
        ("YZ", y, z, (2 * w // 3, 84, w - 1, h - 1)),
    ]
    for label, a, b, (x0, y0, x1, y1) in views:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1)
        cv2.putText(canvas, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        ar = float(np.ptp(a))
        br = float(np.ptp(b))
        if ar < 1e-9 or br < 1e-9:
            continue
        pad = 12
        px = ((a - a.min()) / ar * max((x1 - x0 - 2 * pad), 1) + x0 + pad).astype(int)
        py = ((1.0 - (b - b.min()) / br) * max((y1 - y0 - 2 * pad), 1) + y0 + pad).astype(int)
        inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        canvas[py[inside], px[inside]] = colors[inside]
    return canvas


def _show_pointcloud_interactive(
    rgb_bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    K: np.ndarray,
    title: str,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("[pcd] open3d not found; cannot show interactive view")
        return
    if depth_m is None:
        print("[pcd] no depth available")
        return

    h, w = depth_m.shape[:2]
    if rgb_bgr.shape[:2] != (h, w):
        rgb_bgr = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    if mask is not None:
        mask_bool = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        mask_bool = np.ones((h, w), dtype=bool)

    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        print("[pcd] empty mask")
        return

    z = depth_m[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0) & (~np.isclose(z, 65.535, atol=0.001))
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(xs) == 0:
        print("[pcd] no valid depth points in mask")
        return

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts_x = (xs - cx) * z / fx
    pts_y = (ys - cy) * z / fy
    pts = np.stack([pts_x, pts_y, z], axis=1)
    colors = rgb_bgr[ys, xs][:, ::-1].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[pcd] opening '{title}' - {len(pts):,} points (drag=rotate scroll=zoom Q=close)")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=1200,
        height=800,
        point_show_normal=False,
    )


def _load_mesh(path: Path):
    import open3d as o3d
    import trimesh

    loaded = trimesh.load(Path(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Unable to load mesh: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(loaded.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(loaded.faces))
    try:
        colors = np.asarray(loaded.visual.to_color().vertex_colors)
        if len(colors) == len(loaded.vertices):
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
    except Exception:
        pass
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_vertex_normals()
    return mesh


def _render_pose_scene(
    *,
    active_mesh: Path,
    passive_mesh: Path,
    T_world_active: np.ndarray,
    T_world_passive: np.ndarray,
    out_path: Path,
    camera: dict,
    interactive: bool,
    window_name: str,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("[debug] open3d not found; skipping FP pose scene preview")
        return

    T_rel = np.linalg.inv(np.asarray(T_world_passive, dtype=float)) @ np.asarray(T_world_active, dtype=float)
    passive = _load_mesh(Path(passive_mesh))
    active = _load_mesh(Path(active_mesh))
    active.transform(T_rel)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axes.transform(T_rel)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(window_name=window_name, width=1280, height=720, visible=interactive)
    if not created:
        print(f"[debug] could not create Open3D window for {window_name}; skipping FP pose scene preview")
        return
    try:
        for geom in (passive, active, axes):
            vis.add_geometry(geom)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.08])
        opt.mesh_show_back_face = True
        ctr = vis.get_view_control()
        ctr.set_lookat(list(camera.get("lookat", (0.0, 0.0, 0.0))))
        ctr.set_front(list(camera.get("front", (0.0, -0.8, 0.8))))
        ctr.set_up(list(camera.get("up", (0.0, 0.0, 1.0))))
        ctr.set_zoom(float(camera.get("zoom", 0.9)))
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path), do_render=True)
        if interactive:
            print(f"[debug] {window_name} - close the Open3D window to continue")
            vis.run()
    finally:
        vis.destroy_window()


def _pose_matrix(pose: Any) -> np.ndarray:
    if hasattr(pose, "T_world_object"):
        return np.asarray(pose.T_world_object, dtype=float)
    return np.asarray(pose, dtype=float)


def _log_pose_relation(label: str, active_pose: Any, passive_pose: Any) -> None:
    T_rel = np.linalg.inv(_pose_matrix(passive_pose)) @ _pose_matrix(active_pose)
    xyz = T_rel[:3, 3]
    angle = np.degrees(np.arccos(np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    print(
        f"[scene] {label} active-rel-passive "
        f"xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) rot={angle:.1f} deg"
    )


def _prompt_choice() -> str:
    box: list[str | None] = [None]
    done = threading.Event()

    def _read() -> None:
        try:
            box[0] = input("\n  1-4=redraw box  v1-v4=3D view  Enter=proceed: ").strip()
        except (KeyboardInterrupt, EOFError):
            box[0] = ""
        done.set()

    threading.Thread(target=_read, daemon=True).start()
    old = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        while not done.wait(timeout=0.05):
            cv2.waitKey(1)
    finally:
        signal.signal(signal.SIGINT, old)
    return box[0] or ""


class ProgressLogger:
    def __init__(self, total: int = 12, label: str | None = None):
        self.total = total
        self.label = label

    def step(self, index: int, message: str) -> None:
        prefix = f"[{index}/{self.total}]"
        if self.label:
            print(f"{prefix} {self.label}: {message}")
        else:
            print(f"{prefix} {message}")


@dataclass
class DebugController:
    enabled: bool
    paths: StagePaths
    benchmark: str
    stage_index: int = 0
    stage_count: int = 1

    @property
    def label(self) -> str | None:
        if self.stage_count <= 1:
            return None
        return f"stage {self.stage_index + 1}/{self.stage_count}"

    def show_goal_preview(self, source_bgr: np.ndarray, goal_bgr: np.ndarray) -> None:
        if not self.enabled:
            return
        if not sys.stdin.isatty():
            print("[debug] non-interactive stdin; skipping source/goal preview wait")
            return
        panels = _same_h(_label(source_bgr, "Source"), _label(goal_bgr, "Goal (VLM edit)"))
        _show_and_wait("[2] Source vs Goal", np.hstack(panels))

    def preview_foundationpose_scenes(
        self,
        *,
        active_mesh: Path,
        passive_mesh: Path,
        source_active_pose: Any,
        source_passive_pose: Any,
        goal_active_pose: Any,
        goal_passive_pose: Any,
        camera_views: Any = None,
    ) -> None:
        if not self.enabled:
            return

        views = list(camera_views or ())
        primary_view = views[0] if views else {}
        reverse_view = views[1] if len(views) > 1 else None
        interactive = sys.stdin.isatty()
        if not interactive:
            print("[debug] non-interactive stdin; saving FP pose previews without interactive wait")

        scenes = (
            ("source", source_active_pose, source_passive_pose, "[5] Source Scene - close to continue"),
            ("goal", goal_active_pose, goal_passive_pose, "[5] Goal Scene - close to continue"),
        )
        for scene_name, active_pose, passive_pose, window_name in scenes:
            if active_pose is None or passive_pose is None:
                print(f"[scene] {scene_name}: missing active/passive pose; skipping")
                continue
            _log_pose_relation(f"{scene_name} scene", active_pose, passive_pose)
            old = signal.signal(signal.SIGINT, signal.SIG_DFL)
            try:
                try:
                    _render_pose_scene(
                        active_mesh=active_mesh,
                        passive_mesh=passive_mesh,
                        T_world_active=_pose_matrix(active_pose),
                        T_world_passive=_pose_matrix(passive_pose),
                        out_path=self.paths.debug / f"{scene_name}_fp_scene_axes.png",
                        camera=primary_view,
                        interactive=interactive,
                        window_name=window_name,
                    )
                except Exception as exc:
                    print(f"[debug] FP pose scene preview failed for {scene_name}: {exc}")
                    continue
            finally:
                signal.signal(signal.SIGINT, old)
            if reverse_view is not None:
                try:
                    _render_pose_scene(
                        active_mesh=active_mesh,
                        passive_mesh=passive_mesh,
                        T_world_active=_pose_matrix(active_pose),
                        T_world_passive=_pose_matrix(passive_pose),
                        out_path=self.paths.debug / f"{scene_name}_fp_scene_axes_reverse.png",
                        camera=reverse_view,
                        interactive=False,
                        window_name=f"{window_name} (reverse)",
                    )
                except Exception as exc:
                    print(f"[debug] reverse FP pose scene preview failed for {scene_name}: {exc}")

    def review_masks(
        self,
        *,
        source_bgr: np.ndarray,
        source_depth: Optional[np.ndarray],
        goal_bgr: np.ndarray,
        goal_depth: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        masks: dict[str, ObjectMask],
        mask_dir: Path,
        load_sam2_predictor: Callable[[], Any],
    ) -> dict[str, ObjectMask]:
        if not self.enabled:
            return masks

        mask_dir.mkdir(parents=True, exist_ok=True)
        seg_order = [
            ("source_active", "1 src-active", source_bgr, source_depth),
            ("source_passive", "2 src-passive", source_bgr, source_depth),
            ("goal_active", "3 goal-active", goal_bgr, goal_depth),
            ("goal_passive", "4 goal-passive", goal_bgr, goal_depth),
        ]
        for stem, _, image, _ in seg_order:
            if masks.get(stem) is None:
                masks[stem] = _full_image_mask(stem, image)

        def build_combined() -> np.ndarray:
            overlays = []
            for stem, label, base, _ in seg_order:
                panel = base.copy()
                obj_mask = masks.get(stem)
                if obj_mask is not None and obj_mask.mask is not None:
                    mask = obj_mask.mask
                    if mask.shape[:2] != panel.shape[:2]:
                        mask = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
                    green = np.zeros_like(panel)
                    green[:, :, 1] = mask
                    panel = cv2.addWeighted(panel, 1.0, green, 0.4, 0)
                    x1, y1, x2, y2 = obj_mask.bbox_xyxy
                    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
                panel = cv2.resize(panel, (640, 360))
                info = "none" if obj_mask is None else f"score={obj_mask.score:.2f}"
                cv2.putText(panel, f"{label}  [{info}]", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                overlays.append(panel)

            pointclouds = []
            for stem, label, base, depth in seg_order:
                obj_mask = masks.get(stem)
                mask = obj_mask.mask if obj_mask else None
                panel = _mask_pointcloud_preview_bgr(base, depth, mask, camera_matrix, label)
                panel = cv2.resize(panel, (640, 360))
                cv2.putText(panel, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                pointclouds.append(panel)

            return np.vstack([
                np.vstack([np.hstack(overlays[:2]), np.hstack(overlays[2:])]),
                np.vstack([np.hstack(pointclouds[:2]), np.hstack(pointclouds[2:])]),
            ])

        if not sys.stdin.isatty():
            preview_path = self.paths.debug / f"{self.benchmark}_seg_preview.png"
            cv2.imwrite(str(preview_path), build_combined())
            print(f"[debug] non-interactive stdin; skipping mask review. Preview -> {preview_path}")
            return masks

        sam2_predictor = None
        choices = {"1": 0, "2": 1, "3": 2, "4": 3, "v1": 0, "v2": 1, "v3": 2, "v4": 3}
        print("[seg] review - enter 1-4 to redraw box, v1-v4 for interactive 3D view, Enter to proceed")
        try:
            while True:
                _show("[4] segmentation masks + point cloud inputs", build_combined())
                choice = _prompt_choice()
                if choice == "":
                    _close()
                    break
                idx = choices.get(choice)
                if idx is None:
                    print("  Invalid - enter 1-4, v1-v4, or Enter to proceed")
                    continue

                stem, label, image, depth = seg_order[idx]
                if choice.startswith("v"):
                    obj_mask = masks.get(stem)
                    _show_pointcloud_interactive(
                        image,
                        depth,
                        obj_mask.mask if obj_mask else None,
                        camera_matrix,
                        label,
                    )
                    continue

                win = f"Draw box [{stem}]  ENTER=confirm  C=skip"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                roi = cv2.selectROI(win, image, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(win)
                x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                if w > 0 and h > 0:
                    bbox = (x, y, x + w, y + h)
                    if sam2_predictor is None:
                        sam2_predictor = load_sam2_predictor()
                    sam2_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    sam2_masks, sam2_scores, _ = sam2_predictor.predict(
                        box=np.array([x, y, x + w, y + h], dtype=float),
                        multimask_output=True,
                    )
                    mask = (sam2_masks[int(np.argmax(sam2_scores))] * 255).astype(np.uint8)
                    masks[stem] = ObjectMask(name=stem, mask=mask, bbox_xyxy=bbox, score=1.0)
                    cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), mask)
                    print(f"[seg] {stem}: redrawn bbox={bbox} + SAM2 mask")
                else:
                    masks[stem] = _full_image_mask(stem, image)
                    cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), masks[stem].mask)
                    print(f"[seg] {stem}: using full image mask")
        finally:
            if sam2_predictor is not None:
                del sam2_predictor
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        preview_path = self.paths.debug / f"{self.benchmark}_seg_preview.png"
        cv2.imwrite(str(preview_path), build_combined())
        print(f"[debug] seg preview -> {preview_path}")
        return masks
