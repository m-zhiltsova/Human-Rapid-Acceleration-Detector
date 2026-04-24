import argparse
import cv2
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import onnxruntime as ort
from moge.model.v2 import MoGeModel
from yolox.tracker.byte_tracker import BYTETracker
import time

# ------------------------- YOLOv8Detector -------------------------
class YOLOv8Detector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45, input_size=(1024, 576)):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_width, self.input_height = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def preprocess(self, img):
        self.img_height, self.img_width = img.shape[:2]
        r = min(self.input_width / self.img_width, self.input_height / self.img_height)
        new_w, new_h = int(self.img_width * r), int(self.img_height * r)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        self.scale = r
        self.pad_x, self.pad_y = pad_x, pad_y
        return blob

    def postprocess(self, outputs):
        predictions = outputs[0][0].transpose()
        boxes = []
        confidences = []
        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if class_id == 0 and confidence > self.conf_threshold:
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                x1 = (x1 - self.pad_x) / self.scale
                y1 = (y1 - self.pad_y) / self.scale
                x2 = (x2 - self.pad_x) / self.scale
                y2 = (y2 - self.pad_y) / self.scale
                x1 = max(0, min(x1, self.img_width))
                y1 = max(0, min(y1, self.img_height))
                x2 = max(0, min(x2, self.img_width))
                y2 = max(0, min(y2, self.img_height))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                boxes = [boxes[i] for i in indices.flatten()]
                confidences = [confidences[i] for i in indices.flatten()]
            else:
                boxes, confidences = [], []
        return boxes, confidences

    def detect(self, img):
        blob = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        boxes, confidences = self.postprocess(outputs)
        detections = []
        for box, conf in zip(boxes, confidences):
            detections.append([box[0], box[1], box[2], box[3], conf])
        return np.array(detections) if detections else np.empty((0, 5))

# ------------------------- ByteTracker -------------------------
@dataclass
class ByteTrackerArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 1.6
    min_box_area: float = 10
    mot20: bool = False

class ByteTracker:
    def __init__(self, args=None, frame_rate=30):
        if args is None:
            args = ByteTrackerArgs()
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.frame_id = 0

    def update(self, detections, img_shape):
        self.frame_id += 1
        if detections.shape[0] == 0:
            online_targets = self.tracker.update(np.empty((0, 5)), img_shape, img_shape)
        else:
            online_targets = self.tracker.update(detections, img_shape, img_shape)
        return online_targets

# ------------------------- KalmanTracker -------------------------
class KalmanTracker:
    """Фильтр Калмана для сглаживания траектории и оценки скорости."""
    def __init__(self, dt=1.0, min_speed=0.1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.001
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_update_time = None
        self.dt = dt
        self.min_speed = min_speed

    def update(self, measurement, timestamp):
        if not self.initialized:
            self.kf.statePost = np.array([
                [measurement[0]],
                [measurement[1]],
                [0],
                [0]
            ], np.float32)
            self.initialized = True
            self.last_update_time = timestamp
            return measurement[0], measurement[1], 0.0

        if self.last_update_time is not None:
            dt_real = max(0.01, timestamp - self.last_update_time)
            self.kf.transitionMatrix[0, 2] = dt_real
            self.kf.transitionMatrix[1, 3] = dt_real

        self.kf.predict()
        measured = np.array([[measurement[0]], [measurement[1]]], np.float32)
        self.kf.correct(measured)

        state = self.kf.statePost
        x, z, vx, vz = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        speed = np.sqrt(vx**2 + vz**2)
        if speed < self.min_speed:
            speed = 0.0
        self.last_update_time = timestamp
        return x, z, speed   # <-- обязательно возвращаем три значения
    
    
        

# ------------------------- SpeedEstimator (улучшенный) -------------------------
class SpeedEstimator:
    def __init__(self, fps=30, min_movement=0.2):
        self.fps = fps
        self.min_movement = min_movement
        self.trackers = {}          # track_id -> KalmanTracker
        self.current_speeds = {}    # track_id -> speed (m/s)
        self.last_use_time = {}     # track_id -> timestamp (для очистки)
        self.timeout = 10.0         # секунд без обновлений до удаления трекера

    def _world_coordinates(self, bbox, points_map, mask_map=None):
        x1, y1, x2, y2 = bbox
        h, w = points_map.shape[:2]
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        if mask_map is not None and not mask_map[cy, cx]:
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask_map[ny, nx]:
                        cx, cy = nx, ny
                        break
                else:
                    continue
                break

        point = points_map[cy, cx]
        return point[0], point[1], point[2]  # x, y, z

    def update(self, track_id, bbox, points_map, mask_map, frame_id, timestamp=None):
        if timestamp is None:
            timestamp = frame_id / self.fps

        world_pos = self._world_coordinates(bbox, points_map, mask_map)
        x_raw, y_raw, z_raw = world_pos

        if track_id not in self.trackers:
            self.trackers[track_id] = KalmanTracker(dt=1.0/self.fps, min_speed=self.min_movement)

        tracker = self.trackers[track_id]
        x_f, z_f, speed = tracker.update([x_raw, z_raw], timestamp)

        self.current_speeds[track_id] = speed
        self.last_use_time[track_id] = timestamp

        self._cleanup(timestamp)
        return speed

    def get_speed(self, track_id):
        return self.current_speeds.get(track_id, 0.0)

    def _cleanup(self, current_time):
        to_delete = [tid for tid, t in self.last_use_time.items() if current_time - t > self.timeout]
        for tid in to_delete:
            del self.trackers[tid]
            del self.current_speeds[tid]
            del self.last_use_time[tid]

# ------------------------- Отрисовка -------------------------
def draw_bbox_with_speed(img, bbox, speed=None, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if speed is not None:
        speed_kmh = speed * 3.6
        label = f"{speed_kmh:.1f} km/h"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1 - 2), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# ------------------------- main -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Speed Estimation Pipeline with MoGe")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or RTSP URL")
    parser.add_argument("--output", type=str, default="output.avi", help="Path to output video")
    parser.add_argument("--det_model", type=str, default="yolov8s_576x1024_v3.onnx", help="YOLOv8 ONNX model path")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--moge_interval", type=float, default=1.0, help="MoGe inference interval (seconds)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for MoGe and YOLO (cuda/cpu)")
    parser.add_argument("--min_movement", type=float, default=0.2, help="Minimum movement in meters to consider non-zero speed")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = YOLOv8Detector(args.det_model, conf_thres=args.conf, input_size=(1024, 576))
    tracker = ByteTracker(frame_rate=fps)
    speed_estimator = SpeedEstimator(fps=fps, min_movement=args.min_movement)

    print("Loading MoGe model...")
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    moge_model.eval()
    print("MoGe loaded.")

    frame_id = 0
    moge_interval_frames = int(fps * args.moge_interval)
    last_moge_frame = -moge_interval_frames
    points = None
    mask = None

    print("Processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        detections = detector.detect(frame)
        online_targets = tracker.update(detections, (height, width))

        # Вызов MoGe с заданным интервалом
        if frame_id - last_moge_frame >= moge_interval_frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            with torch.no_grad():
                output = moge_model.infer(input_tensor)
            points = output["points"].cpu().numpy()
            mask = output["mask"].cpu().numpy()
            last_moge_frame = frame_id

        # Обновление скорости и отрисовка (только если есть данные MoGe)
        if points is not None:
            for t in online_targets:
                tlwh = t.tlwh
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                bbox = (x1, y1, x2, y2)
                track_id = t.track_id

                speed = speed_estimator.update(track_id, bbox, points, mask, frame_id)
                draw_bbox_with_speed(frame, bbox, speed)

        out.write(frame)

        if frame_id % 100 == 0:
            print(f"Processed {frame_id} frames")

    cap.release()
    out.release()
    print(f"Done. Output saved to {args.output}")

if __name__ == "__main__":
    main()