# Теперь можно импортировать
import sys
from pathlib import Path

# Абсолютный путь к папке с main.py
BASE = Path(__file__).resolve().parent

# Добавляем папку Depth-Anything-V2 в список поиска модулей
sys.path.insert(0, str(BASE / "Depth-Anything-V2"))
import cv2
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import onnxruntime as ort
import torch
import time
import argparse
from collections import defaultdict, deque

from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from depth_anything_v2.dpt import DepthAnythingV2


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESH   = 0.35
NMS_THRESH    = 0.45
INPUT_SIZE    = 640          # YOLOv8 input side
PERSON_CLS_ID = 0

# Depth → real-world scale calibration
# Depth Anything outputs *relative* inverse-depth. We convert with:
#   real_distance_m = DEPTH_SCALE / depth_value
# Tune DEPTH_SCALE for your camera / scene (default ≈ 5 m at mid-scene depth)
DEPTH_SCALE   = 5.0

# Speed smoothing: keep last N frames per track
SPEED_HISTORY = 8

# Colour palette (BGR)
BOX_COLOR     = (0, 230, 118)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_COLOR    = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8 ONNX DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class YOLOv8Detector:
    def __init__(self, model_path: str):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape  # [1,3,H,W]

    def preprocess(self, frame: np.ndarray):
        """Letterbox + normalise → (1,3,640,640) float32"""
        h, w = frame.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_top  = (INPUT_SIZE - nh) // 2
        pad_left = (INPUT_SIZE - nw) // 2
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]          # HWC → 1CHW
        return blob, scale, pad_top, pad_left

    def postprocess(self, output, scale, pad_top, pad_left,
                    orig_h, orig_w):
        """
        YOLOv8 output: [1, 84, 8400]  (cx, cy, w, h, cls0…cls83)
        Returns list of [x1, y1, x2, y2, conf, cls]
        """
        preds = output[0][0]          # (84, 8400)
        preds = preds.T               # (8400, 84)

        boxes   = preds[:, :4]        # cx,cy,w,h  (letterbox coords)
        scores  = preds[:, 4:]        # class scores (no obj score in v8)
        cls_ids = scores.argmax(axis=1)
        confs   = scores.max(axis=1)

        # Keep only PERSON class above threshold
        mask = (cls_ids == PERSON_CLS_ID) & (confs >= CONF_THRESH)
        boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]

        if len(boxes) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2  (letterbox space)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Remove padding, undo scale → original image coords
        x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h)

        # NMS
        idxs = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, x2-x1, y2-y1], axis=1).tolist(),
            confs.tolist(), CONF_THRESH, NMS_THRESH
        )
        if len(idxs) == 0:
            return []
        idxs = idxs.flatten()
        return np.stack([x1[idxs], y1[idxs],
                         x2[idxs], y2[idxs],
                         confs[idxs], cls_ids[idxs].astype(float)], axis=1)

    def detect(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad_top, pad_left = self.preprocess(frame)
        outputs = self.sess.run(None, {self.input_name: blob})
        return self.postprocess(outputs, scale, pad_top, pad_left,
                                orig_h, orig_w)


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH ESTIMATOR  (Depth Anything V2 – Small)
# ─────────────────────────────────────────────────────────────────────────────
class DepthEstimator:
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, encoder: str = "vitl",
                 checkpoint: str = "checkpoints/depth_anything_v2_vitl.pth"):
        cfg = self.MODEL_CONFIGS[encoder]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DepthAnythingV2(**cfg)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Returns depth map (H×W float32), higher = closer by convention here."""
        depth = self.model.infer_image(frame_rgb)   # returns np array
        return depth.astype(np.float32)

    def sample_depth(self, depth_map: np.ndarray,
                     cx: int, cy: int, radius: int = 5) -> float:
        """
        Sample median depth in a small patch around (cx, cy).
        Converts relative depth → approximate metres using DEPTH_SCALE.
        """
        h, w = depth_map.shape
        x1, y1 = max(0, cx-radius), max(0, cy-radius)
        x2, y2 = min(w, cx+radius), min(h, cy+radius)
        patch = depth_map[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        med = float(np.median(patch))
        if med < 1e-6:
            return 0.0
        return DEPTH_SCALE / med          # metres


# ─────────────────────────────────────────────────────────────────────────────
# SPEED TRACKER  (wraps ByteTrack + distance history)
# ─────────────────────────────────────────────────────────────────────────────
class SpeedTracker:
    def __init__(self, fps: float):
        self.fps = fps
        args = SimpleNamespace(
            track_thresh=0.45,
            track_buffer=30,
            match_thresh=0.8,
            mot20=False,
        )
        self.tracker  = BYTETracker(args, frame_rate=int(fps))
        # track_id → deque of (timestamp_s, distance_m)
        self.history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=SPEED_HISTORY + 1)
        )
        self.speeds: dict[int, float] = {}

    def update(self, detections, depth_map: np.ndarray,
               frame_idx: int) -> list:
        """
        detections: Nx6 array [x1,y1,x2,y2,conf,cls]
        Returns list of (track_id, x1,y1,x2,y2, speed_kmh)
        """
        t = frame_idx / self.fps

        if len(detections) == 0:
            self.tracker.update(
                np.empty((0, 5), dtype=np.float32),
                [frame_idx, frame_idx], [1, 1]
            )
            return []

        dets_bt = np.hstack([
            detections[:, :4],
            detections[:, 4:5]
        ]).astype(np.float32)

        img_info = [depth_map.shape[0], depth_map.shape[1]]
        img_size = img_info

        online_targets = self.tracker.update(dets_bt, img_info, img_size)

        results = []
        for t_obj in online_targets:
            tlwh = t_obj.tlwh
            tid  = t_obj.track_id
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            x2 = int(tlwh[0] + tlwh[2])
            y2 = int(tlwh[1] + tlwh[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            
            dist_m = self._get_distance(depth_map, cx, cy)
            ts     = frame_idx / self.fps

            self.history[tid].append((ts, dist_m))
            speed = self._calc_speed(tid)
            self.speeds[tid] = speed

            results.append((tid, x1, y1, x2, y2, speed))

        return results

    def _get_distance(self, depth_map, cx, cy):
        estimator_ref = _depth_estimator_ref   # module-level ref set in main
        return estimator_ref.sample_depth(depth_map, cx, cy)

    def _calc_speed(self, tid: int) -> float:
        hist = self.history[tid]
        if len(hist) < 2:
            return 0.0
        t0, d0 = hist[0]
        t1, d1 = hist[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return 0.0
        # 3-D displacement approximation: we only have depth change (z-axis)
        # For in-plane motion, use pixel displacement scaled by depth.
        # Here we use absolute depth difference as proxy distance change.
        dd = abs(d1 - d0)           # metres along camera axis
        speed_ms  = dd / dt
        speed_kmh = speed_ms * 3.6
        return round(speed_kmh, 1)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def draw_results(frame: np.ndarray, results: list) -> np.ndarray:
    vis = frame.copy()
    for (tid, x1, y1, x2, y2, speed) in results:
        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Label background + text
        label = f"ID:{tid}  {speed} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                      TEXT_BG_COLOR, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1,
                    cv2.LINE_AA)

        # Centre dot
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL REF used inside SpeedTracker._get_distance
# ─────────────────────────────────────────────────────────────────────────────
_depth_estimator_ref: DepthEstimator = None   # set in main()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run(input_path: str, output_path: str,
        onnx_model: str, depth_encoder: str,
        depth_checkpoint: str):

    global _depth_estimator_ref

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Video: {width}x{height}  {fps:.1f} fps  {total} frames")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8 ONNX detector …")
    detector = YOLOv8Detector(onnx_model)

    print(f"[INFO] Loading Depth Anything V2 ({depth_encoder}) …")
    depth_est = DepthEstimator(encoder=depth_encoder,
                               checkpoint=depth_checkpoint)
    _depth_estimator_ref = depth_est

    print("[INFO] Initialising ByteTrack …")
    speed_tracker = SpeedTracker(fps=fps)

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_idx = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Detect
        dets = detector.detect(frame)
        if not isinstance(dets, np.ndarray) or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        # 2. Depth
        depth_map = depth_est.infer(frame_rgb)

        # 3. Track + speed
        results = speed_tracker.update(dets, depth_map, frame_idx)

        # 4. Draw
        vis = draw_results(frame, results)

        # HUD
        elapsed = time.time() - t_start
        proc_fps = (frame_idx + 1) / max(elapsed, 1e-6)
        cv2.putText(vis,
                    f"Frame {frame_idx}/{total}  |  {proc_fps:.1f} fps",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(vis)

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{total}  tracks={len(results)}"
                  f"  proc={proc_fps:.1f} fps")

        frame_idx += 1

    cap.release()
    out.release()
    elapsed = time.time() - t_start
    print(f"\n[DONE] Processed {frame_idx} frames in {elapsed:.1f}s → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People speed estimator")
    parser.add_argument("--input",  default="video.mp4")
    parser.add_argument("--output", default="out.mp4")
    parser.add_argument("--model",  default="yolov8n_best.onnx",
                        help="Path to YOLOv8 ONNX model")
    parser.add_argument("--depth-encoder", default="vitl",
                        choices=["vits", "vitb", "vitl"],
                        help="Depth Anything V2 encoder size")
    parser.add_argument("--depth-checkpoint",
                        default="checkpoints/depth_anything_v2_vitl.pth",
                        help="Path to Depth Anything V2 checkpoint")
    args = parser.parse_args()

    run(
        input_path      = args.input,
        output_path     = args.output,
        onnx_model      = args.model,
        depth_encoder   = args.depth_encoder,
        depth_checkpoint= args.depth_checkpoint,
    )