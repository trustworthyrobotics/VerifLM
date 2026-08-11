import pyrealsense2 as rs
import cv2
import numpy as np
from pathlib import Path
import os
import json

class ArucoBoard:
    def __init__(self, markers_x=2, markers_y=2, marker_length_m=0.115, marker_separation=0.015):
        self.markers_x = markers_x
        self.markers_y = markers_y
        self.marker_length_m = marker_length_m
        self.marker_separation=marker_separation
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.GridBoard((markers_x, markers_y), marker_length_m, marker_separation, self.dictionary)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)

    def detect(self, bgr): 
        corners, ids, _ = self.detector.detectMarkers(bgr)
        return corners, ids

    def get_extrinsics(self, bgr, K, dist_coeffs, corners, ids):
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs)
        if not success:
            raise RuntimeError("[camera]: solvePnP failed!")
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape((3, )) # id array of 3 elements
        T_cam_board = np.eye(4)
        T_cam_board[0:3, 0:3] = R
        T_cam_board[0:3, 3] = t
        return T_cam_board
        

class Camera():
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.started = False
        self.K = None
        self.dist_coeffs = None
        self.depth_scale_m = 1 # multiply raw-depth readings by this scaling factor get get depth in meters
        self.width = width
        self.height = height
        self.fps = fps
        self.T_cam_board = None

    def start(self):
        if self.started:
            return
        self.started = True

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        # get intrinsics
        profile = self.pipeline.start(self.config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()
        self.K = np.array([
            [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
            [0.0, self.intrinsics.fy, self.intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(self.intrinsics.coeffs, dtype=np.float64)

        # get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale_m = float(depth_sensor.get_depth_scale())

        # warm up camera
        got_frame = False
        for i in range(30):
            frames = self.pipeline.wait_for_frames(1000)
            if frames:
                got_frame = True
        if not got_frame:
            raise RuntimeError("[camera]: no frame received during warmup.")
        print("[camera]: started")

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
            print("[camera]: stopped")

    def capture(self, render=False, save=False, calculate_extrinsics=False, out_dir=None, prefix="source"):
        frames = None
        for _ in range(5):
            frames = self.pipeline.wait_for_frames(1000)
            frames = self.align.process(frames)
            if not frames:
                continue
            else:
                break

        if not frames:
            return

        # read frame
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = depth_raw * self.depth_scale_m
        depth_m[(depth_m <= 0) | (~np.isfinite(depth_m))] = np.nan

        # try to detect aruco board
        if calculate_extrinsics:
            board = ArucoBoard()
            corners, ids = board.detect(bgr)
            if ids is None or len(ids) == 0:
                raise RuntimeError("[camera]: failed to detect arucoboard")
            self.T_cam_board = board.get_extrinsics(bgr, self.K, self.dist_coeffs, corners, ids)

            if out_dir is not None:
                extrinsics_path = out_dir/f"{prefix}_extrinsics.npy"
                np.save(extrinsics_path, self.T_cam_board)
        else:
            extrinsics_path = out_dir/f"{prefix}_extrinsics.npy"
            self.T_cam_board = np.load(extrinsics_path) if extrinsics_path.exists() else None

        if render:
            min_depth_m = 0.2
            max_depth_m = 3.0
            valid = np.isfinite(depth_m)
            depth_scaled = np.zeros(depth_m.shape, dtype=np.uint8)
            normalized = (255 * (max_depth_m - depth_m[valid]) / (max_depth_m - min_depth_m))
            depth_scaled[valid] = np.clip(normalized, 0, 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_scaled,cv2.COLORMAP_TURBO) 

            combined = cv2.hconcat([bgr, depth_vis])
            cv2.imshow("Camera", combined)
            cv2.waitKey(1)
        if save:
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True, parents=True)
            rgb_path = out_dir / f"{prefix}_rgb.png"
            depth_path = out_dir / f"{prefix}_depth_m.npy"
            meta_path = out_dir / f"{prefix}_camera_meta.json"
            metadata = {
                'rgb_path': str(rgb_path), 
                'depth_path': str(depth_path),
                'K': self.K.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                # 'T_cam_board': T_cam_board.tolist() if isinstance(T_cam_board, np.ndarray) else None,
            }

            cv2.imwrite(rgb_path, bgr)
            np.save(depth_path, depth_m)
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"[camera]: saved {rgb_path}, and {depth_path}")

    def get_intrinsics(self):
        # [[fx, 0, cx],
        #  [0, fy, cy],
        #  [0, 0, 1]]
        return self.K
    
    def get_extrinsics(self):
        return self.T_cam_board

