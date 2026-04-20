import argparse
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
import onnxruntime as ort
from moge.model.v2 import MoGeModel
from yolox.tracker.byte_tracker import BYTETracker, STrack

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

# ------------------------- SpeedEstimator (улучшенный) -------------------------
class SpeedEstimator:
    def __init__(self, fps=30, speed_interval_sec=2.0, pos_smooth_window=5, speed_smooth_alpha=0.3):
        self.fps = fps
        self.speed_interval_frames = int(fps * speed_interval_sec)
        self.pos_smooth_window = pos_smooth_window
        self.speed_smooth_alpha = speed_smooth_alpha

        # История сглаженных позиций: deque из (frame_id, x, z)
        self.history = defaultdict(lambda: deque(maxlen=self.pos_smooth_window * 2))
        # Сырые позиции для сглаживания координат: deque из (x, z)
        self.raw_positions = defaultdict(lambda: deque(maxlen=pos_smooth_window))
        # Последнее вычисление скорости
        self.last_speed_calc_frame = defaultdict(int)
        # Текущие скорости (после фильтрации)
        self.current_speeds = {}

    def _world_coordinates(self, bbox, points_map, mask_map=None):
        x1, y1, x2, y2 = bbox
        h, w = points_map.shape[:2]
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        if mask_map is not None and not mask_map[cy, cx]:
            # Поиск ближайшего валидного пикселя в окне 5x5
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
        return point[0], point[1], point[2]  # (x, y, z)

    def update(self, track_id, bbox, points_map, mask_map, frame_id):
        # Получаем сырую позицию
        world_pos = self._world_coordinates(bbox, points_map, mask_map)
        # Сохраняем в буфер для сглаживания
        self.raw_positions[track_id].append((world_pos[0], world_pos[2]))

        # Вычисляем сглаженную позицию (среднее по окну)
        if len(self.raw_positions[track_id]) >= self.pos_smooth_window:
            avg_x = np.mean([p[0] for p in self.raw_positions[track_id]])
            avg_z = np.mean([p[1] for p in self.raw_positions[track_id]])
            smooth_pos = (avg_x, avg_z)
        else:
            smooth_pos = (world_pos[0], world_pos[2])

        # Сохраняем сглаженную позицию в историю
        self.history[track_id].append((frame_id, smooth_pos[0], smooth_pos[1]))

        # Проверяем, пора ли пересчитать скорость
        if frame_id - self.last_speed_calc_frame[track_id] >= self.speed_interval_frames:
            speed = self._compute_speed(track_id, frame_id)
            # Экспоненциальное сглаживание скорости
            prev_speed = self.current_speeds.get(track_id)
            if prev_speed is not None:
                speed = self.speed_smooth_alpha * speed + (1 - self.speed_smooth_alpha) * prev_speed
            self.current_speeds[track_id] = speed
            self.last_speed_calc_frame[track_id] = frame_id
            return speed
        return None

    def _compute_speed(self, track_id, current_frame_id):
        hist = list(self.history[track_id])
        if len(hist) < 2:
            return 0.0

        # Берём точки, попадающие в интервал
        min_frame_id = current_frame_id - self.speed_interval_frames
        # Находим самую раннюю точку в интервале
        early_idx = 0
        for i, (fid, _, _) in enumerate(hist):
            if fid >= min_frame_id:
                early_idx = i
                break
        early_frame, early_x, early_z = hist[early_idx]
        late_frame, late_x, late_z = hist[-1]

        if late_frame == early_frame:
            return 0.0

        dt = (late_frame - early_frame) / self.fps
        if dt <= 0:
            return 0.0

        distance = np.sqrt((late_x - early_x)**2 + (late_z - early_z)**2)
        speed = distance / dt  # м/с
        return speed

    def get_speed(self, track_id):
        return self.current_speeds.get(track_id, None)

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
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output.avi", help="Path to output video")
    parser.add_argument("--det_model", type=str, default="yolov8s_576x1024_v3.onnx", help="YOLOv8 ONNX model path")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--speed_interval", type=float, default=2.0, help="Speed calculation interval (seconds)")
    parser.add_argument("--moge_interval", type=float, default=1.0, help="MoGe inference interval (seconds)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for MoGe and YOLO (cuda/cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = YOLOv8Detector(args.det_model, conf_thres=args.conf, input_size=(1024, 576))
    tracker = ByteTracker(frame_rate=fps)
    speed_estimator = SpeedEstimator(fps=fps, speed_interval_sec=args.speed_interval)

    print("Loading MoGe model...")
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    moge_model.eval()
    print("MoGe loaded.")

    frame_id = 0
    moge_interval_frames = int(fps * args.moge_interval)
    last_moge_frame = -moge_interval_frames
    points = None
    mask = None

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Детекция и трекинг
        detections = detector.detect(frame)
        online_targets = tracker.update(detections, (height, width))

        # Вызов MoGe с интервалом
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

                speed_updated = speed_estimator.update(track_id, bbox, points, mask, frame_id)
                # Отображаем последнюю известную скорость
                display_speed = speed_estimator.get_speed(track_id)
                draw_bbox_with_speed(frame, bbox, display_speed)

        out.write(frame)

        if frame_id % 100 == 0:
            print(f"Processed {frame_id} / {total_frames} frames")

    cap.release()
    out.release()
    print(f"Done. Output saved to {args.output}")

if __name__ == "__main__":
    main()