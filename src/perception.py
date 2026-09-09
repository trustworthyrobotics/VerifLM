import threading
import signal
import torch
import time
import subprocess
import numpy as np
import json
import cv2
import trimesh
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class DepthEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()

    def predict(self, input_paths: list=[], output_paths=None):
        # Jointly infer the depth of both images result in more consistent geometry and depth scale
        images = load_and_preprocess_images(input_paths).to(self.device)
        with torch.no_grad():
            pred = self.model(images) # torch.Size([1,N,H,W,1])
        depth_raw = pred["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()  # torch.Size([N,H,W])
        depth_raw_resized = []

        for i in range(len(depth_raw)):
            images = cv2.imread(str(input_paths[i]))
            depth = cv2.resize(depth_raw[i], (images.shape[1], images.shape[0]), interpolation=cv2.INTER_LINEAR)
            depth_raw_resized.append(depth)

        if output_paths:
            for i in range(len(output_paths)):
                np.save(output_paths[i], depth_raw_resized[i])
        return depth_raw

    def correct(self, source_depth_paths, source_mask_paths, edited_depth_path, output_path):
        # Compare VGGT-estimated depth against groundtruth depth to calculate D'=s*D+t
        # source_depth_paths[0] = camera depth, source_depth_paths[1] = vggt depth
        # source_mask_paths[0] = active object mask, source_mask_paths[1] = passive object mask
        # We only compare the depth maps of active and passive objects, since these are the most trustworthy regions

        source_true_depth = np.load(source_depth_paths[0])
        source_vggt_depth = np.load(source_depth_paths[1])
        edited_vggt_depth = np.load(edited_depth_path)

        # source_active_mask = cv2.imread(str(source_mask_paths[0]), cv2.IMREAD_GRAYSCALE) > 0 # converts into an array of booleans
        # source_passive_mask = cv2.imread(str(source_mask_paths[1]), cv2.IMREAD_GRAYSCALE) > 0
        # valid = source_active_mask | source_passive_mask
        valid = (source_true_depth >= 0.15) & (source_true_depth <= 2.0) & (source_vggt_depth > 0)

        source_true_depth_filtered = source_true_depth[valid]
        source_vggt_depth_filtered = source_vggt_depth[valid]

        valid = (np.isfinite(source_true_depth_filtered) & np.isfinite(source_vggt_depth_filtered) &
                 (source_true_depth_filtered > 0) & (source_vggt_depth_filtered > 0))

        source_true_depth_filtered = source_true_depth_filtered[valid]
        source_vggt_depth_filtered = source_vggt_depth_filtered[valid]

        s, t = np.polyfit(source_vggt_depth_filtered, source_true_depth_filtered, 1)
        print(f"[perception]: D'={s} * D + {t}")
        edited_vggt_depth_m = s * edited_vggt_depth + t

        np.save(output_path, edited_vggt_depth_m.astype(np.float32))
        return edited_vggt_depth_m, s, t

class BBoxer:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.model = load_model(str(self.repo_path / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                                str(self.repo_path / "weights/groundingdino_swint_ogc.pth"))

    def predict(self, input_paths, prompts, output_path=None, loop=False):
        bbox_list = [None] * len(input_paths)

        for i, (path, prompt), in enumerate(zip(input_paths, prompts)):
            bgr = cv2.imread(str(path))
            _, image = load_image(str(path))

            bboxes, scores, _ = predict(self.model, image, prompt, box_threshold=0.35, text_threshold=0.25)

            if len(bboxes) == 0: # failed prediction
                continue
                
            h, w = bgr.shape[:2]
            cx, cy, bw, bh = bboxes[scores.argmax()].tolist()

            bbox_list[i] = (int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)) # (xmin, ymin, xmax, ymax)

        if loop:
            self.review(input_paths, bbox_list)

        if output_path: 
            Path(output_path).write_text(json.dumps(bbox_list, indent=4))

        return bbox_list
    
    @staticmethod
    def _prompt_choice():
        box = [None]
        done = threading.Event()

        def _read():
            try:
                box[0] = input("\nRedraw box [1-4], or Enter to accept: ").strip()
            except (KeyboardInterrupt, EOFError):
                box[0] = ""
            done.set()

        threading.Thread(target=_read, daemon=True).start()
        old = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            while not done.wait(0.05):
                cv2.waitKey(1)
        finally:
            signal.signal(signal.SIGINT, old)

        return box[0] or ""

    def review(self, input_paths, bbox_list):
        while True:
            panels = []
            failed = False

            for i, path in enumerate(input_paths):
                image = cv2.imread(str(path))
                bbox = bbox_list[i]

                if bbox is not None:
                    cv2.rectangle(image, bbox[:2], bbox[2:], (0, 255, 0), 2)
                    status = str(bbox)
                else:
                    failed = True
                    status = "DETECTION FAILED - redraw"

                cv2.putText(image, f"{i + 1}: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                panels.append(cv2.resize(image, (640, 360)))

            preview = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
            cv2.imshow("BBoxes", preview)

            choice = self._prompt_choice()
            if not choice:
                if failed:
                    print("[perception]: please redraw failed detections!")
                    continue
                break

            if choice not in {"1","2","3","4"}:
                continue    

            choice_i = int(choice) - 1
            image = cv2.imread(str(input_paths[choice_i]))
            x, y, w, h = map(int, cv2.selectROI(f"Draw box {choice}", image, fromCenter=False, showCrosshair=True))
            cv2.destroyWindow(f"Draw box {choice}")

            if w > 0 and h > 0:
                bbox_list[choice_i] = (x, y, x+w, y+h)

        cv2.destroyAllWindows()

            
class Segmenter:
    def __init__(self, repo_path):
        repo_path = Path(repo_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        build_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(repo_path / "checkpoints/sam2.1_hiera_large.pt"), device=self.device)
        self.model = SAM2ImagePredictor(build_model)

    def predict(self, input_paths, bboxes, output_paths):
        masks = []

        for path, bbox in zip(input_paths, bboxes):
            image = cv2.imread(str(path))
            self.model.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            pred_masks, scores, _ = self.model.predict(box=bbox, multimask_output=False) # assumes (px_min, py_min, px_max, py_max)
            mask = pred_masks[0].astype(np.uint8) * 255
            masks.append(mask)

        if output_paths:
            for mask, path in zip(masks, output_paths):
                cv2.imwrite(str(path), mask)
        return masks

class PoseEstimator:
    def __init__(self, repo_path, container="foundationpose_realsense"):
        self.repo_path = Path(repo_path)
        self.container = container
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.container], capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError(f"[perception]: container '{container}' does not exist")
        if r.stdout.strip() != "true": subprocess.run(["docker", "start", container], check=True, stdout=subprocess.DEVNULL)

    def _cleanup(self, timeout=5):
        subprocess.run(
            ["docker", "exec", self.container, "pkill", "-9", "-f", "python"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
        )

        end = time.time() + timeout
        while time.time() < end:
            result = subprocess.run(
                ["docker", "exec", self.container, "pgrep", "-f", "python"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                return
            time.sleep(0.5)
        print("[perception]: FP cleaned up successfully")

    @staticmethod
    def _debug_projection(rgb_path, depth_path, mask_path, mesh_path, K, T, output_path):
        rgb, depth = cv2.imread(str(rgb_path)), np.load(depth_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        ys, xs = np.where(mask)
        valid = mask & np.isfinite(depth) & (depth > 0)
        mesh = trimesh.load(mesh_path, force="scene", process=False)
        if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        points = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
        points = points[points[:, 2] > 1e-6]
        uv = points[:, :2] / points[:, 2, None] @ K[:2, :2].T + K[:2, 2]
        px, py = uv[:, 0], uv[:, 1]
        mb = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        pb = np.array([px.min(), py.min(), px.max(), py.max()]).round().astype(int)
        inter = max(0, min(mb[2], pb[2]) - max(mb[0], pb[0])) * max(0, min(mb[3], pb[3]) - max(mb[1], pb[1]))
        iou = inter / (np.prod(mb[2:] - mb[:2]) + np.prod(pb[2:] - pb[:2]) - inter + 1e-9)
        d = depth[valid]
        print(f"[perception]: mask center=({xs.mean():.1f},{ys.mean():.1f}) bbox={mb.tolist()} | "
              f"FP origin=({K[0,0]*T[0,3]/T[2,3]+K[0,2]:.1f},{K[1,1]*T[1,3]/T[2,3]+K[1,2]:.1f}) "
              f"mesh bbox={pb.tolist()} bbox-IoU={iou:.3f} | depth p1/50/99={np.percentile(d, [1,50,99]).round(3)}")
        cv2.drawContours(rgb, [cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0, 255, 0), 2)
        cv2.rectangle(rgb, tuple(pb[:2]), tuple(pb[2:]), (0, 0, 255), 2)
        cv2.imwrite(str(output_path), rgb)

    def predict(self, rgb_path, depth_path, mask_path, mesh_path, K, output_dir, output_name):
        output_dir = Path(output_dir).resolve()
        debug_dir = output_dir / Path(output_name).stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker", "exec", self.container,
            "python", str(self.repo_path / "run_frame.py"),
            "--mesh_file", str(Path(mesh_path).resolve()),
            "--color_file", str(Path(rgb_path).resolve()),
            "--depth_file", str(Path(depth_path).resolve()),
            "--mask_file", str(Path(mask_path).resolve()),
            "--fx", str(float(K[0, 0])),
            "--fy", str(float(K[1, 1])),
            "--cx", str(float(K[0, 2])),
            "--cy", str(float(K[1, 2])),
            "--debug", "1",
            "--debug_dir", str(debug_dir),
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        finally:
            self._cleanup()
            print("[perception]: FP finished")

        if result.returncode != 0:
            error = "\n".join(result.stdout.decode(errors="replace").splitlines()[-8:])
            raise RuntimeError(f"[perception]: FP failed with error: \n{error}")
        
        pose_path = debug_dir / "ob_in_cam" / "000000.txt"
        if not pose_path.exists():
            raise RuntimeError(f"[perception]: FP produced no pose")

        T_cam_object = np.loadtxt(pose_path)
        self._debug_projection(rgb_path, depth_path, mask_path, mesh_path, K, T_cam_object,
                               debug_dir / "projection_check.png")
        np.save(output_dir/output_name, T_cam_object)

        return T_cam_object
