import torch
import numpy as np
import json
import cv2
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
        images = load_and_preprocess_images(input_paths).to(self.device)
        with torch.no_grad():
            pred = self.model(images) # torch.Size([1,N,H,W,1])
        depth_raw = pred["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()  # torch.Size([N,H,W])

        if output_paths:
            for i in range(len(output_paths)):
                np.save(output_paths[i], depth_raw[i])
        return depth_raw

    def correct(self, depth_paths, mask_paths, output_path):
        # Compares the depth maps of passive object in the source image against the edited image, then calculate a scaling factor
        # paths[0] = source, paths[1] = edited
        source_depth = np.load(depth_paths[0])
        edited_depth = np.load(depth_paths[1])

        source_mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE) > 0
        edited_mask = cv2.imread(str(mask_paths[1]), cv2.IMREAD_GRAYSCALE) > 0

        source_values = source_depth[source_mask]
        edited_values = edited_depth[edited_mask]

        source_values = source_values[np.isfinite(source_values) & (source_values > 0)]
        edited_values = edited_values[np.isfinite(edited_values) & (edited_values > 0)]

        scale = np.median(source_values) / np.median(edited_values)
        corrected = edited_depth * scale

        np.save(output_path, corrected.astype(np.float32))
        return corrected, scale


class BBoxer:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.model = load_model(str(self.repo_path / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                                str(self.repo_path / "weights/groundingdino_swint_ogc.pth"))

    def predict(self, input_paths, prompts, output_path=None, loop=False):
        output_json = {}

        for path, prompt, in zip(input_paths, prompts):
            bgr = cv2.imread(str(path))
            _, image = load_image(str(path))

            bboxes, scores, _ = predict(self.model, image, prompt, box_threshold=0.35, text_threshold=0.25)

            if len(bboxes) == 0: # failed prediction
                output_json[prompt] = None
                continue
                
            h, w = bgr.shape[:2]
            cx, cy, bw, bh = bboxes[scores.argmax()].tolist()

            output_json[prompt] = (int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)) # (xmin, ymin, xmax, ymax)

        if loop:
            self.review(input_paths, output_json)

        if output_path: 
            Path(output_path).write_text(json.dumps(output_json))

        return output_json

    def review(self, input_paths, output_json):
        # Wait for user to redraw bboxes 
        keys = list(output_json)

        while True:
            panels = []
            failed = False

            for i, (path, key) in enumerate(zip(input_paths, keys)):
                image = cv2.imread(str(path))
                bbox = output_json[key]

                if bbox is not None:
                    cv2.rectangle(image, bbox[:2], bbox[2:], (0, 255, 0), 2)
                    status = str(bbox)
                else:
                    failed = True
                    status = "DETECTION FAILED - redraw"

                cv2.putText(image, f"{i+1}: {key} | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                panels.append(cv2.resize(image, (640,360)))

            preview = np.vstack([np.hstack(panels[0:2]), np.hstack(panels[2:])])
            cv2.imshow("BBoxes", preview)
            cv2.waitKey(1)

            choice = input("Redraw box [1-4], or Enter to accept: ").strip()
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
                output_json[keys[choice_i]] = (x, y, x+w, y+h)

        cv2.destroyAllWindows()
            
class Segmenter:
    def __init__(self, repo_path):
        repo_path = Path(repo_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        build_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(repo_path / "checkpoints/sam2.1_hiera_large.pt"), device=self.device)
        self.model = SAM2ImagePredictor(build_model)

    def predict(self, input_paths, bboxes, output_paths):
        masks = []

        for path, bbox in zip(input_paths, bboxes.values()):
            image = cv2.imread(str(path))
            self.model.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            pred_masks, scores, _ = self.model.predict(box=bbox, multimask_output=False) # assumes (px_min, py_min, px_max, py_max)
            mask = pred_masks[0].astype(np.uint8) * 255
            masks.append(mask)

        if output_paths:
            for mask, path in zip(masks, output_paths):
                cv2.imwrite(str(path), mask)
        return masks

