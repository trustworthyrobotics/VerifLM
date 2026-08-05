from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal
import json
import time

import cv2
import numpy as np

CameraType = Literal["gemini", "realsense", "d435", "realsense_d435"]


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class RGBDFrame:
    rgb_bgr: np.ndarray
    depth_m: Optional[np.ndarray]
    intrinsics: CameraIntrinsics
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    timestamp: float
    T_camera_to_board: Optional[np.ndarray] = None
    T_board_to_world: Optional[np.ndarray] = None
    board_vis_bgr: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        return int(self.rgb_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb_bgr.shape[1])


@dataclass
class BoardConfig:
    dictionary: str = "DICT_4X4_50"
    markers_x: int = 2
    markers_y: int = 2
    marker_length_m: float = 0.115
    marker_separation_m: float = 0.015
    axis_length_m: float = 0.05


@dataclass
class RawRGBD:
    rgb_bgr: np.ndarray
    depth_m: Optional[np.ndarray]
    intrinsics: CameraIntrinsics
    dist_coeffs: np.ndarray
    timestamp: float


def create_board(cfg: BoardConfig):
    dict_id = getattr(cv2.aruco, cfg.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.GridBoard(
        (cfg.markers_x, cfg.markers_y),
        cfg.marker_length_m,
        cfg.marker_separation_m,
        dictionary,
    )
    return dictionary, board


def recenter_board_points(obj_points: np.ndarray, cfg: BoardConfig) -> np.ndarray:
    board_w = cfg.markers_x * cfg.marker_length_m + (cfg.markers_x - 1) * cfg.marker_separation_m
    board_h = cfg.markers_y * cfg.marker_length_m + (cfg.markers_y - 1) * cfg.marker_separation_m

    pts = obj_points.reshape(-1, 3).astype(np.float64)
    pts[:, 0] -= board_w / 2.0
    pts[:, 1] -= board_h / 2.0
    return pts.reshape(obj_points.shape)


def board_to_camera_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = tvec.reshape(3)
    return T


def camera_to_board_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return np.linalg.inv(board_to_camera_transform(rvec, tvec))


def camera_to_world_transform(
    T_camera_to_board: Optional[np.ndarray],
    T_board_to_world: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if T_camera_to_board is None:
        return None
    board_to_world = np.eye(4, dtype=np.float64) if T_board_to_world is None else np.asarray(T_board_to_world, dtype=np.float64)
    return board_to_world @ np.asarray(T_camera_to_board, dtype=np.float64)


def _as_transform(T: Optional[np.ndarray]) -> np.ndarray:
    if T is None:
        return np.eye(4, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got shape {T.shape}.")
    return T.copy()


class ArucoBoardTracker:
    def __init__(self, cfg: BoardConfig) -> None:
        self.cfg = cfg
        self.dictionary, self.board = create_board(cfg)
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            cv2.aruco.DetectorParameters(),
        )

    def estimate_pose(
        self,
        frame_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        vis = frame_bgr.copy()
        corners, ids, _ = self.detector.detectMarkers(frame_bgr)

        if ids is None or len(ids) == 0:
            return vis, None, None, None

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        if obj_points is None or img_points is None or len(obj_points) < 4:
            return vis, None, None, ids

        obj_points = recenter_board_points(obj_points, self.cfg)
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if not ok:
            return vis, None, None, ids

        cv2.drawFrameAxes(
            vis,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            self.cfg.axis_length_m,
        )
        return vis, rvec, tvec, ids


def _load_orbbec_sdk():
    try:
        import pyorbbecsdk2 as ob  # type: ignore
        return ob
    except ImportError:
        try:
            import pyorbbecsdk as ob  # type: ignore
            return ob
        except ImportError as exc:
            raise ImportError(
                "Could not import pyorbbecsdk2 or pyorbbecsdk. Install the Orbbec SDK "
                "before using Camera(camera_type='gemini'), or run without --use-camera."
            ) from exc


def _load_realsense_sdk():
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ImportError as exc:
        raise ImportError(
            "Could not import pyrealsense2. Install Intel RealSense SDK / pyrealsense2 "
            "before using Camera(camera_type='realsense')."
        ) from exc


class _GeminiBackend:
    def __init__(
        self,
        timeout_ms: int,
        enable_depth: bool,
        align_depth_to_color: bool,
        warmup_frames: int,
        depth_unit_fallback_m: float,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color
        self.warmup_frames = warmup_frames
        self.depth_unit_fallback_m = depth_unit_fallback_m

        self.ob = _load_orbbec_sdk()
        self.pipeline = self.ob.Pipeline()
        self.config = self.ob.Config()

        self._started = False
        self._depth_enabled = False
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._color_format = None
        self._depth_scale_m = depth_unit_fallback_m

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("Camera intrinsics unavailable. Call start() first.")
        return self._intrinsics

    def start(self) -> None:
        if self._started:
            return

        self._enable_color_stream()
        if self.enable_depth:
            self._enable_depth_stream()
        if self._depth_enabled and self.align_depth_to_color:
            self._try_enable_depth_to_color_alignment()

        self.pipeline.start(self.config)
        self._started = True
        time.sleep(0.5)  # allow sensor firmware to start streaming before polling

        color_frame, depth_frame = self._warmup_until_color()
        if color_frame is None:
            self.stop()
            raise RuntimeError(
                "Camera started, but no color frame was received after warmup. "
                "Try enable_depth=False to debug color-only streaming."
            )

        self._read_intrinsics_from_color_frame(color_frame)
        self._read_color_format(color_frame)
        if self._depth_enabled and depth_frame is not None:
            self._depth_scale_m = self._get_depth_scale_m(depth_frame)

        print("[camera:gemini] started")
        print(f"[camera:gemini] resolution: {self.intrinsics.width}x{self.intrinsics.height}")
        print(f"[camera:gemini] K:\n{self.intrinsics.K}")
        print(f"[camera:gemini] depth_enabled={self._depth_enabled}, depth_scale={self._depth_scale_m:.8f} m/unit")

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
            self._started = False
            print("[camera:gemini] stopped")

    def capture_raw(self, retries: int) -> RawRGBD:
        if not self._started:
            raise RuntimeError("Camera is not started. Call start() first.")

        last_depth_m = None  # carry forward depth from depth-only frames
        for attempt in range(retries):
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
            if frames is None:
                continue

            # Cache depth even when color is absent so a later color frame can use it.
            if self._depth_enabled:
                depth_frame = self._safe_get_depth_frame(frames)
                if depth_frame is not None:
                    d = self._decode_depth_frame(depth_frame, target_shape=None)
                    if d is not None:
                        last_depth_m = d

            color_frame = self._safe_get_color_frame(frames)
            if color_frame is None:
                continue

            rgb_bgr = self._decode_color_frame(color_frame)
            depth_m = None
            if self._depth_enabled:
                # Try same-frame depth first; fall back to cached depth.
                depth_frame = self._safe_get_depth_frame(frames)
                if depth_frame is not None:
                    depth_m = self._decode_depth_frame(depth_frame, target_shape=rgb_bgr.shape[:2])
                if depth_m is None and last_depth_m is not None:
                    h, w = rgb_bgr.shape[:2]
                    depth_m = cv2.resize(last_depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
                if depth_m is None and attempt < retries - 1:
                    print(f"[camera:gemini] depth not ready (attempt {attempt + 1}/{retries}), retrying…")
                    continue

            return RawRGBD(
                rgb_bgr=rgb_bgr,
                depth_m=depth_m,
                intrinsics=self.intrinsics,
                dist_coeffs=np.zeros((5, 1), dtype=np.float64),
                timestamp=time.time(),
            )

        raise RuntimeError("Failed to capture a frame with both color and depth.")

    def _enable_color_stream(self) -> None:
        profiles = self.pipeline.get_stream_profile_list(self.ob.OBSensorType.COLOR_SENSOR)
        self.config.enable_stream(profiles.get_default_video_stream_profile())

    def _enable_depth_stream(self) -> None:
        try:
            profiles = self.pipeline.get_stream_profile_list(self.ob.OBSensorType.DEPTH_SENSOR)
            self.config.enable_stream(profiles.get_default_video_stream_profile())
            self._depth_enabled = True
        except Exception as exc:
            print(f"[camera:gemini] depth stream unavailable: {exc}")
            self._depth_enabled = False

    def _try_enable_depth_to_color_alignment(self) -> None:
        try:
            if not hasattr(self.config, "set_align_mode") or not hasattr(self.ob, "OBAlignMode"):
                return
            for name in ("SW_MODE", "ALIGN_D2C_SW_MODE", "HW_MODE", "ALIGN_D2C_HW_MODE"):
                if hasattr(self.ob.OBAlignMode, name):
                    self.config.set_align_mode(getattr(self.ob.OBAlignMode, name))
                    print(f"[camera:gemini] enabled depth-to-color alignment: {name}")
                    return
        except Exception as exc:
            print(f"[camera:gemini] depth-to-color alignment unavailable: {exc}")

    def _warmup_until_color(self):
        last_color = None
        last_depth = None
        for i in range(self.warmup_frames):
            frames = self.pipeline.wait_for_frames(2000)
            if frames is None:
                print(f"[camera:gemini] warmup {i}: no frames")
                continue
            color_frame = self._safe_get_color_frame(frames)
            depth_frame = self._safe_get_depth_frame(frames)
            print(f"[camera:gemini] warmup {i}: color={'yes' if color_frame is not None else 'no'} depth={'yes' if depth_frame is not None else 'no'}")
            if color_frame is not None:
                last_color = color_frame
                if depth_frame is not None:
                    last_depth = depth_frame
                # Keep warming up until depth also arrives when depth is enabled.
                if not self._depth_enabled or last_depth is not None:
                    break
        return last_color, last_depth

    @staticmethod
    def _safe_get_color_frame(frames):
        try:
            return frames.get_color_frame()
        except Exception:
            return None

    @staticmethod
    def _safe_get_depth_frame(frames):
        try:
            return frames.get_depth_frame()
        except Exception:
            return None

    def _read_intrinsics_from_color_frame(self, color_frame) -> None:
        profile = color_frame.get_stream_profile().as_video_stream_profile()
        intr = profile.get_intrinsic()
        self._intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
        )

    def _read_color_format(self, color_frame) -> None:
        try:
            self._color_format = color_frame.get_format()
            print(f"[camera:gemini] color format: {self._color_format}")
        except Exception:
            self._color_format = None

    def _decode_color_frame(self, color_frame) -> np.ndarray:
        vf = color_frame.as_video_frame()
        w, h = int(vf.get_width()), int(vf.get_height())
        raw = np.frombuffer(vf.get_data(), dtype=np.uint8)
        n = raw.size

        if n == h * w * 3:
            return raw.reshape((h, w, 3)).copy()
        if n == h * w:
            return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)
        if n == h * w * 2:
            return cv2.cvtColor(raw.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUY2)

        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        raise RuntimeError(f"Unsupported color frame layout: size={n}, width={w}, height={h}, format={self._color_format}")

    def _decode_depth_frame(self, depth_frame, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        vf = depth_frame.as_video_frame()
        w, h = int(vf.get_width()), int(vf.get_height())
        # bytes() copies the C++ buffer before the frame goes out of scope
        raw = np.frombuffer(bytes(vf.get_data()), dtype=np.uint16).reshape(h, w)
        depth_m = raw.astype(np.float32) * float(self._depth_scale_m)
        depth_m[raw == np.iinfo(np.uint16).max] = np.nan

        if target_shape is not None and depth_m.shape != target_shape:
            target_h, target_w = target_shape
            depth_m = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        depth_m[depth_m <= 0] = np.nan
        depth_m[~np.isfinite(depth_m)] = np.nan
        return depth_m

    def _get_depth_scale_m(self, depth_frame) -> float:
        for attr in ("get_depth_scale", "get_value_scale"):
            if hasattr(depth_frame, attr):
                try:
                    scale = float(getattr(depth_frame, attr)())
                    if scale > 0:
                        # Orbbec reports millimeters per raw depth unit.
                        return scale / 1000.0
                except Exception:
                    pass
        return float(self.depth_unit_fallback_m)


class _RealSenseD435Backend:
    def __init__(
        self,
        timeout_ms: int,
        enable_depth: bool,
        align_depth_to_color: bool,
        warmup_frames: int,
        color_width: int,
        color_height: int,
        depth_width: int,
        depth_height: int,
        fps: int,
        serial: Optional[str] = None,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color
        self.warmup_frames = warmup_frames
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.serial = serial

        self.rs = _load_realsense_sdk()
        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()
        if serial:
            self.config.enable_device(serial)
        self.align = self.rs.align(self.rs.stream.color) if align_depth_to_color else None

        self._started = False
        self._depth_enabled = False
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._depth_scale_m = 0.001

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("Camera intrinsics unavailable. Call start() first.")
        return self._intrinsics

    def start(self) -> None:
        if self._started:
            return

        ctx = self.rs.context()
        devices = ctx.query_devices()
        target = None
        for i in range(devices.size()):
            d = devices[i]
            if self.serial is None or d.get_info(self.rs.camera_info.serial_number) == self.serial:
                target = d
                break
        if target is not None:
            print(f"[camera:realsense_d435] hardware reset ...")
            target.hardware_reset()
            time.sleep(2)

        self.config.enable_stream(
            self.rs.stream.color,
            self.color_width,
            self.color_height,
            self.rs.format.bgr8,
            self.fps,
        )
        if self.enable_depth:
            self.config.enable_stream(
                self.rs.stream.depth,
                self.depth_width,
                self.depth_height,
                self.rs.format.z16,
                self.fps,
            )
            self._depth_enabled = True

        profile = self.pipeline.start(self.config)
        self._started = True

        color_profile = profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
        )

        if self._depth_enabled:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale_m = float(depth_sensor.get_depth_scale())

        self._warmup()
        print("[camera:realsense_d435] started")
        print(f"[camera:realsense_d435] K:\n{self.intrinsics.K}")
        print(f"[camera:realsense_d435] depth_enabled={self._depth_enabled}, depth_scale={self._depth_scale_m:.8f} m/unit")

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
            self._started = False
            print("[camera:realsense_d435] stopped")

    def capture_raw(self, retries: int) -> RawRGBD:
        if not self._started:
            raise RuntimeError("Camera is not started. Call start() first.")

        for _ in range(retries):
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
            if not frames:
                continue
            if self._depth_enabled and self.align is not None:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data()).copy()
            depth_m = None
            if self._depth_enabled:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    depth_m = depth_raw.astype(np.float32) * float(self._depth_scale_m)
                    if depth_m.shape != rgb_bgr.shape[:2]:
                        depth_m = cv2.resize(depth_m, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                    depth_m[depth_m <= 0] = np.nan
                    depth_m[~np.isfinite(depth_m)] = np.nan

            return RawRGBD(
                rgb_bgr=rgb_bgr,
                depth_m=depth_m,
                intrinsics=self.intrinsics,
                dist_coeffs=np.zeros((5, 1), dtype=np.float64),
                timestamp=time.time(),
            )

        raise RuntimeError("Failed to capture a color frame.")

    def _warmup(self) -> None:
        got_first = False
        for i in range(self.warmup_frames):
            frames = self.pipeline.wait_for_frames(5000)
            if frames:
                if not got_first:
                    print(f"[camera:realsense_d435] first frame at warmup {i}, settling auto-exposure ...")
                    got_first = True
            else:
                if not got_first:
                    print(f"[camera:realsense_d435] warmup {i}: no frames")
        if not got_first:
            raise RuntimeError("RealSense: no frames received during warmup.")


class Camera:
    """Single RGB-D camera wrapper for Orbbec Gemini or Intel RealSense D435.

    Use one parameter to choose the hardware:
        Camera(camera_type="gemini")
        Camera(camera_type="realsense")  # also accepts "d435" or "realsense_d435"

    Output convention:
      rgb_bgr: OpenCV BGR uint8 image, H x W x 3
      depth_m: float32 depth in meters, H x W, NaN for invalid pixels
      camera_matrix: 3 x 3 intrinsics matrix
      T_camera_to_board: estimated from the ArUco board when detected
      T_board_to_world: user-defined constant board/world offset
    """

    def __init__(
        self,
        camera_type: CameraType = "gemini",
        timeout_ms: int = 100,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        warmup_frames: int = 30,
        depth_unit_fallback_m: float = 0.001,
        enable_board_tracking: bool = True,
        board_config: Optional[BoardConfig] = None,
        T_board_to_world: Optional[np.ndarray] = None,
        # RealSense defaults. Gemini keeps its previous default stream profiles.
        color_width: int = 1280,
        color_height: int = 720,
        depth_width: int = 1280,
        depth_height: int = 720,
        fps: int = 30,
        serial: Optional[str] = None,
    ) -> None:
        self.camera_type = self._normalize_camera_type(camera_type)
        self.timeout_ms = timeout_ms
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color
        self.warmup_frames = warmup_frames
        self.depth_unit_fallback_m = depth_unit_fallback_m
        self.enable_board_tracking = enable_board_tracking

        if self.camera_type == "gemini":
            self._backend = _GeminiBackend(
                timeout_ms=timeout_ms,
                enable_depth=enable_depth,
                align_depth_to_color=align_depth_to_color,
                warmup_frames=warmup_frames,
                depth_unit_fallback_m=depth_unit_fallback_m,
            )
        else:
            self._backend = _RealSenseD435Backend(
                timeout_ms=timeout_ms,
                enable_depth=enable_depth,
                align_depth_to_color=align_depth_to_color,
                warmup_frames=warmup_frames,
                color_width=color_width,
                color_height=color_height,
                depth_width=depth_width,
                depth_height=depth_height,
                fps=fps,
                serial=serial,
            )

        self._board_tracker = self._create_board_tracker(board_config or BoardConfig())
        self.T_camera_to_board: Optional[np.ndarray] = None
        self.T_board_to_world = _as_transform(T_board_to_world)
        self.last_board_vis_bgr: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_camera_type(camera_type: str) -> str:
        name = camera_type.lower().strip().replace("-", "_")
        if name == "gemini":
            return "gemini"
        if name in {"realsense", "d435", "realsense_d435", "rs_d435"}:
            return "realsense_d435"
        raise ValueError("camera_type must be 'gemini', 'realsense', 'd435', or 'realsense_d435'.")

    def start(self) -> None:
        self._backend.start()

    def stop(self) -> None:
        self._backend.stop()

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._backend.intrinsics

    def capture(self, retries: int = 30) -> RGBDFrame:
        raw = self._backend.capture_raw(retries=retries)
        camera_matrix = raw.intrinsics.K.copy()
        dist_coeffs = raw.dist_coeffs

        board_vis_bgr = None
        T_camera_to_board = None if self.T_camera_to_board is None else self.T_camera_to_board.copy()
        if self._board_tracker is not None:
            board_vis_bgr, rvec, tvec, _ = self._board_tracker.estimate_pose(raw.rgb_bgr, camera_matrix, dist_coeffs)
            if rvec is not None and tvec is not None:
                self.T_camera_to_board = camera_to_board_transform(rvec, tvec)
                T_camera_to_board = self.T_camera_to_board.copy()
            self.last_board_vis_bgr = board_vis_bgr

        return RGBDFrame(
            rgb_bgr=raw.rgb_bgr,
            depth_m=raw.depth_m,
            intrinsics=raw.intrinsics,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            timestamp=raw.timestamp,
            T_camera_to_board=T_camera_to_board,
            T_board_to_world=self.T_board_to_world.copy(),
            board_vis_bgr=board_vis_bgr,
        )

    def save_frame(self, frame: RGBDFrame, out_dir: str | Path, prefix: str = "source") -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{prefix}_{stamp}"

        rgb_path = out_dir / f"{stem}_rgb.png"
        meta_path = out_dir / f"{stem}_meta.json"
        cv2.imwrite(str(rgb_path), frame.rgb_bgr)

        depth_path = None
        depth_vis_path = None
        depth_stats = None
        if frame.depth_m is not None:
            depth_path = out_dir / f"{stem}_depth_mm.png"
            depth_vis_path = out_dir / f"{stem}_depth_vis.png"
            valid_depth = (
                np.isfinite(frame.depth_m)
                & (frame.depth_m > 0.02)
                & (frame.depth_m < 10.0)
            )
            valid_values = frame.depth_m[valid_depth]
            depth_stats = {
                "valid_pixels": int(valid_values.size),
                "min_m": float(valid_values.min()) if valid_values.size else None,
                "max_m": float(valid_values.max()) if valid_values.size else None,
            }
            print(f"[camera] depth stats: {depth_stats}")
            depth_mm = np.nan_to_num(frame.depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(np.round(depth_mm * 1000.0), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_mm)
            cv2.imwrite(str(depth_vis_path), make_depth_vis(frame.depth_m))

        meta = {
            "camera_type": self.camera_type,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path) if depth_path else None,
            "depth_vis_path": str(depth_vis_path) if depth_vis_path else None,
            "depth_unit": "millimeters",
            "depth_stats": depth_stats,
            "depth_aligned_to_rgb": bool(self.align_depth_to_color),
            "timestamp": frame.timestamp,
            "image_width": frame.width,
            "image_height": frame.height,
            "camera_matrix": frame.camera_matrix.tolist(),
            "dist_coeffs": frame.dist_coeffs.tolist(),
            "T_camera_to_board": frame.T_camera_to_board.tolist() if frame.T_camera_to_board is not None else None,
            "T_board_to_world": frame.T_board_to_world.tolist() if frame.T_board_to_world is not None else None,
            "intrinsics": {
                "width": frame.intrinsics.width,
                "height": frame.intrinsics.height,
                "fx": frame.intrinsics.fx,
                "fy": frame.intrinsics.fy,
                "cx": frame.intrinsics.cx,
                "cy": frame.intrinsics.cy,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[camera] saved RGB-D frame under {out_dir}")
        return meta

    def _create_board_tracker(self, cfg: BoardConfig) -> Optional[ArucoBoardTracker]:
        if not self.enable_board_tracking:
            return None
        if not hasattr(cv2, "aruco"):
            print("[camera] OpenCV ArUco module unavailable; board tracking disabled")
            return None
        try:
            return ArucoBoardTracker(cfg)
        except Exception as exc:
            print(f"[camera] ArUco board tracking unavailable: {exc}")
            return None


def make_depth_vis(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    clean = np.zeros_like(depth_m, dtype=np.float32)
    clean[valid] = depth_m[valid]
    vis = cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-type", default="gemini", choices=["gemini", "realsense", "d435", "realsense_d435"])
    parser.add_argument("--out-dir", default="assets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    with Camera(camera_type=args.camera_type, enable_depth=True, align_depth_to_color=True) as cam:
        while True:
            frame = cam.capture()
            color_vis = frame.board_vis_bgr if frame.board_vis_bgr is not None else frame.rgb_bgr
            if frame.depth_m is not None:
                show = np.hstack([color_vis, make_depth_vis(frame.depth_m)])
            else:
                show = color_vis
            cv2.imshow(f"{args.camera_type} RGB-D | s save | q quit", show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cam.save_frame(frame, out_dir, prefix="source")
            elif key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()


def load_extrinsic(path: str | Path) -> np.ndarray:
    """Load a 4x4 camera-to-board transform from extrinsics/*.txt."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        text = line.strip()
        if text.startswith("#"):
            continue
        if "4x4" in text and ("T_cam_in_board" in text or "T_cam_in_world" in text):
            return np.asarray(
                [[float(value) for value in lines[index + row].split()] for row in range(1, 5)],
                dtype=np.float64,
            )
    raise ValueError(f"Could not parse camera extrinsic from {path}")


def save_extrinsic(path: str | Path, T_camera_to_board: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["T_cam_in_board 4x4"]
    lines.extend("  " + "  ".join(f"{value: .8f}" for value in row) for row in T_camera_to_board)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def capture_source(args) -> RGBDFrame:
    """Capture one source RGB-D frame and attach the persisted camera extrinsic."""
    from pipeline.core import extrinsic_path

    path = extrinsic_path(args.camera_type)
    with Camera(
        camera_type=args.camera_type,
        serial=args.camera_serial,
        enable_depth=True,
        align_depth_to_color=True,
        warmup_frames=60,
    ) as camera:
        if path.exists():
            camera.T_camera_to_board = load_extrinsic(path)
        frame = camera.capture()
        if frame.T_camera_to_board is None:
            raise RuntimeError(
                f"No extrinsic at {path} and the ArUco board was not detected in the captured frame."
            )
        if not path.exists():
            save_extrinsic(path, frame.T_camera_to_board)
        frame.T_board_to_world = np.eye(4, dtype=np.float64)
        return frame
