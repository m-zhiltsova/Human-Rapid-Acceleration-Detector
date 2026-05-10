import os, sys, cv2, json, time, argparse, subprocess
# Ваш токен HF
#os.environ["HF_TOKEN"] = "hf_hash"
from collections import deque
import numpy as np

if not hasattr(np, 'float'):
    np.float = float

import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass

# --- Умный импорт BoTSORT для поддержки любой версии boxmot ---
try:
    from boxmot import BotSort
    BoTSORT_CLASS = BotSort
except ImportError:
    from boxmot import BoTSORT
    BoTSORT_CLASS = BoTSORT


# ------------------------------------------------------------------ #
#  Inline depth estimation (optional)                                 #
# ------------------------------------------------------------------ #
def run_depth_estimator(empty_frame, camera_id, device,
                        clahe=0.0, denoise=0.0, sharpen=0.0):
    script = Path(__file__).parent / 'depth_estimator.py'
    if not script.exists():
        print(f"[ERROR] depth_estimator.py not found next to this script ({script})")
        sys.exit(1)
    cmd = [
        sys.executable, str(script),
        '--input',     str(empty_frame),
        '--camera_id', str(camera_id),
        '--device',    str(device),
    ]
    if clahe   > 0: cmd += ['--clahe',   str(clahe)]
    if denoise > 0: cmd += ['--denoise', str(denoise)]
    if sharpen > 0: cmd += ['--sharpen', str(sharpen)]
    print(f"[depth] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ #
#  Image filters                                                      #
# ------------------------------------------------------------------ #
def apply_filters(frame, clahe_clip=0.0, denoise_h=0.0, sharpen_amount=0.0):
    if clahe_clip > 0:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_Lab2BGR)
    if denoise_h > 0:
        d = 7 if denoise_h >= 50 else 5
        frame = cv2.bilateralFilter(frame, d=d, sigmaColor=denoise_h, sigmaSpace=denoise_h)
    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(frame, (0,0), 3)
        frame = cv2.addWeighted(frame, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    return frame


# ------------------------------------------------------------------ #
#  YOLOv8 Detector                                                    #
# ------------------------------------------------------------------ #
class YOLOv8Detector:
    PERSON_CLASS_ID = 0
    NMS_THRESHOLD = 0.45
    MODEL_INPUT_SHAPE = (576, 1024)

    def __init__(self, model_path, conf_thres=0.45, device='cuda'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"[YOLO] Providers: {self.session.get_providers()}")
        self.conf_threshold = conf_thres
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _letterbox(self, img, new_shape=MODEL_INPUT_SHAPE, color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2; dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _preprocess(self, frame):
        img, ratio, pad = self._letterbox(frame, self.MODEL_INPUT_SHAPE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_norm, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        return img_tensor, ratio, pad

    def _postprocess(self, output, orig_shape, ratio, pad):
        detections = output[0]
        if detections.shape[1] == 0:
            return np.empty((0, 5))
        if detections.shape[0] == 5:
            boxes = detections[:4, :]
            scores = detections[4:5, :]
            person_scores = scores[0, :]
        else:
            boxes = detections[:4, :]
            scores = detections[4:, :]
            if scores.shape[0] == 0:
                return np.empty((0, 5))
            person_scores = scores[self.PERSON_CLASS_ID, :]
        valid_idx = np.where(person_scores > self.conf_threshold)[0]
        if len(valid_idx) == 0:
            return np.empty((0, 5))
        h_input, w_input = self.MODEL_INPUT_SHAPE
        boxes_xyxy, confidences = [], []
        for idx in valid_idx:
            cx, cy, w, h = boxes[:, idx]
            x1 = max(0, min(cx - w/2, w_input))
            y1 = max(0, min(cy - h/2, h_input))
            x2 = max(0, min(cx + w/2, w_input))
            y2 = max(0, min(cy + h/2, h_input))
            boxes_xyxy.append([x1, y1, x2, y2])
            confidences.append(person_scores[idx])
        boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_xyxy]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.NMS_THRESHOLD)
        if len(indices) == 0:
            return np.empty((0, 5))
        r, (dw, dh) = ratio, pad
        final_dets = []
        for i in indices.flatten():
            x1, y1, w, h = boxes_xywh[i]
            x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
            w /= r;              h /= r
            x2, y2 = x1 + w, y1 + h
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            final_dets.append([x1, y1, x2, y2, confidences[i]])
        return np.array(final_dets)

    def detect(self, img):
        img_tensor, ratio, pad = self._preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: img_tensor})
        return self._postprocess(outputs[0], img.shape[:2], ratio, pad)


# ------------------------------------------------------------------ #
#  BoTSORT Tracker (ИСПРАВЛЕННЫЙ)                                     #
# ------------------------------------------------------------------ #
class BoTSORTTracker:
    def __init__(self, frame_rate=30, device='cuda'):
        #  osnet_x0_25_msmt17.pt ->osnet_x1_0_msmt17.pt
        self.tracker = BoTSORT_CLASS(
            reid_weights=Path('osnet_x1_0_msmt17.pt'),
            device=device,
            half=False,
            track_high_thresh=0.5,   # Игнорируем слабые боксы-фантомы
            track_low_thresh=0.1,    # Оставляем классический порог для 2-го прохода (перекрытия)
            new_track_thresh=0.6,    # Жесткий порог: новый ID даем только уверенным детекциям
            track_buffer=240,        # Память 8 сек (при 30fps), чтобы трек жил во время перекрытия
            match_thresh=0.8,        # Строгий IoU матчинг
        )
        self._prev_ids: set[int] = set()

    def update(self, dets: np.ndarray, frame: np.ndarray) -> list:
        # Проверка на пустые детекции
        if dets.shape[0] == 0:
            self.tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            self._prev_ids = set()
            return []
            
        # Защита от битых/бесконечных боксов
        valid_dets = dets[np.isfinite(dets).all(axis=1)]
        if valid_dets.shape[0] == 0:
            return []

        cls_col    = np.zeros((valid_dets.shape[0], 1), dtype=np.float32)
        dets_w_cls = np.hstack((valid_dets, cls_col))
        
        # Обновление трекера
        tracks = self.tracker.update(dets_w_cls, frame)

        class TrackResult:
            __slots__ = ('tlwh', 'track_id')
            def __init__(self, x1, y1, x2, y2, tid):
                self.tlwh     = [float(x1), float(y1),
                                 float(x2-x1), float(y2-y1)]
                self.track_id = int(tid)

        results = []
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                # Фильтр вырожденных bbox, из-за которых скачут координаты
                if (t[2]-t[0]) < 10 or (t[3]-t[1]) < 10:
                    continue
                results.append(TrackResult(t[0], t[1], t[2], t[3], t[4]))

        self._prev_ids = {r.track_id for r in results}
        return results


# ------------------------------------------------------------------ #
#  KalmanTracker — state: [X, Z, Vx, Vz]                              #
# ------------------------------------------------------------------ #
class KalmanTracker:
    PROC_POS  = 0.002
    PROC_VEL  = 0.05
    MEAS_NOISE = 0.7

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov   = np.diag([
            self.PROC_POS, self.PROC_POS,
            self.PROC_VEL, self.PROC_VEL,
        ]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (self.MEAS_NOISE ** 2)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self.initialized = False
        self.last_time   = None

    def update(self, x: float, z: float, t: float):
        if not self.initialized:
            self.kf.statePost = np.array([[x],[z],[0.0],[0.0]], np.float32)
            self.initialized  = True
            self.last_time    = t
            return x, z, 0.0, 0.0
        dt = max(1e-3, t - self.last_time)
        self.last_time = t
        F = np.eye(4, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        self.kf.transitionMatrix = F
        qa = 0.5
        dt2 = dt * dt
        dt3 = dt2 * dt
        self.kf.processNoiseCov = np.diag([
            0.5 * qa * dt3, 0.5 * qa * dt3, qa * dt, qa * dt
        ]).astype(np.float32)
        self.kf.predict()
        self.kf.correct(np.array([[x],[z]], np.float32))
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])


# ------------------------------------------------------------------ #
#  SpeedEstimator                                                    #
# ------------------------------------------------------------------ #
MAX_PIX_JUMP   = 40
MAX_HEIGHT_CHG = 0.40
MIN_OBS        = 12
ABS_MAX_SPEED  = 15.0
WINDOW_SEC     = 5.0
EMA_ALPHA      = 0.10
DISPLAY_INTERVAL = 4

class SpeedEstimator:
    def __init__(self, fps=30, window_size=WINDOW_SEC,
                 smooth_alpha=EMA_ALPHA, display_interval=DISPLAY_INTERVAL):
        self.fps              = fps
        self.window_size      = window_size
        self.smooth_alpha     = smooth_alpha
        self.display_interval = display_interval
        self.timeout          = 10.0

        self.trackers          = {}
        self.history           = {}
        self.current_speed     = {}
        self.display_speed     = {}
        self.last_display_time = {}
        self.last_time         = {}
        self.last_pixel        = {}
        self.last_bbox_h       = {}

        self.points_map  = None
        self.mask        = None
        self.K           = None
        self.px_scale_x  = 1.0
        self.px_scale_y  = 1.0

    def load_depth_map(self, camera_id: str,
                       video_w: int = 0, video_h: int = 0,
                       apply_calibration: bool = True):
        cfg = Path('cfg') / camera_id
        self.points_map = np.load(cfg / 'points_map.npy')
        self.mask       = np.load(cfg / 'mask.npy')
        with open(cfg / 'meta.json') as f:
            meta = json.load(f)

        dm_h, dm_w = self.points_map.shape[:2]
        src_w = float(video_w  if video_w  > 0 else meta.get('original_width',  dm_w))
        src_h = float(video_h  if video_h  > 0 else meta.get('original_height', dm_h))
        self.px_scale_x = dm_w / src_w
        self.px_scale_y = dm_h / src_h

        meta_w = float(meta.get('original_width',  0))
        meta_h = float(meta.get('original_height', 0))
        if meta_w > 0 and meta_h > 0 and (meta_w != src_w or meta_h != src_h):
            print(f"[SpeedEstimator] WARNING: empty_frame was {int(meta_w)}×{int(meta_h)} "
                  f"but video is {int(src_w)}×{int(src_h)}. "
                  f"Depth map may not match the scene geometry!")

        K_path = cfg / 'intrinsics.npy'
        if K_path.exists():
            self.K = np.load(K_path)
        else:
            fl = max(dm_h, dm_w) * 1.2
            self.K = np.array([[fl, 0, dm_w/2],
                                [0, fl, dm_h/2],
                                [0,  0,      1]], dtype=np.float32)

        self._apply_depth_calibration(cfg, apply_calibration)
        print(f"[SpeedEstimator] depth map {dm_w}×{dm_h}, "
              f"video {int(src_w)}×{int(src_h)}, "
              f"scale x={self.px_scale_x:.4f} y={self.px_scale_y:.4f}")

    def _apply_depth_calibration(self, cfg_dir, apply):
        if not apply:
            return
        cal_file = cfg_dir / 'calibration.json'
        if not cal_file.exists():
            return
        try:
            with open(cal_file) as f:
                calib = json.load(f)
            if calib.get('type') == 'polynomial':
                poly = np.poly1d(calib['coefficients'])
                print(f"[SpeedEstimator] Applying calibration: {poly}")
                Z = self.points_map[..., 2]
                K = poly(Z)
                self.points_map = self.points_map * K[..., np.newaxis]
        except Exception as e:
            print(f"[SpeedEstimator] Calibration error: {e}")

    def _sample_point(self, cx_orig: float, cy_orig: float):
        h, w = self.points_map.shape[:2]
        cx = cx_orig * self.px_scale_x
        cy = cy_orig * self.px_scale_y
        cx_i = int(round(np.clip(cx, 0, w - 1)))
        cy_i = int(round(np.clip(cy, 0, h - 1)))
        if self.mask[cy_i, cx_i]:
            pt = self.points_map[cy_i, cx_i]
            return float(pt[0]), float(pt[2])
        LAT_VID = 8
        VRT_VID = 5
        lat = max(1, int(round(LAT_VID * self.px_scale_x)))
        vrt = max(1, int(round(VRT_VID * self.px_scale_y)))
        for dy in range(0, vrt + 1):
            for dx in range(0, lat + 1):
                for sx, sy in ([(0,0),(dx,0),(-dx,0),(0,-dy),(dx,-dy),(-dx,-dy)]
                                if dy > 0 else [(0,0),(dx,0),(-dx,0)]):
                    nx, ny = cx_i + sx, cy_i + sy
                    if 0 <= nx < w and 0 <= ny < h and self.mask[ny, nx]:
                        pt = self.points_map[ny, nx]
                        return float(pt[0]), float(pt[2])
        pt = self.points_map[cy_i, cx_i]
        return float(pt[0]), float(pt[2])

    def update(self, track_id: int, bbox, t: float) -> float:
        x1, y1, x2, y2 = bbox
        cx_px  = (x1 + x2) / 2.0
        cy_px = float(y2) - 0.07 * (y2 - y1)
        bbox_h = float(y2 - y1)

        if track_id not in self.trackers:
            self.trackers[track_id]          = KalmanTracker()
            raw_x, raw_z = self._sample_point(cx_px, cy_px)
            self.trackers[track_id].update(raw_x, raw_z, t)
            self.history[track_id]           = deque()
            self.history[track_id].append((t, raw_x, raw_z))
            self.current_speed[track_id]     = 0.0
            self.display_speed[track_id]     = 0.0
            self.last_display_time[track_id] = t - self.display_interval
            self.last_time[track_id]         = t
            self.last_pixel[track_id]        = (cx_px, cy_px)
            self.last_bbox_h[track_id]       = bbox_h
            self._cleanup(t)
            return 0.0

        last_cx, last_cy = self.last_pixel[track_id]
        pix_jump = np.sqrt((cx_px - last_cx)**2 + (cy_px - last_cy)**2)
        bbox_w = float(x2 - x1)
        jump_thresh = np.clip(bbox_w * 0.4, 15.0, 60.0)
        if pix_jump > jump_thresh:
            self.last_time[track_id] = t
            self.last_pixel[track_id] = (cx_px, cy_px)
            self.last_bbox_h[track_id] = bbox_h
            self._cleanup(t)
            return self.display_speed[track_id]

        prev_h = self.last_bbox_h[track_id]
        if prev_h > 0 and abs(bbox_h - prev_h) / prev_h > MAX_HEIGHT_CHG:
            self.last_time[track_id] = t
            self.last_pixel[track_id] = (cx_px, cy_px)
            self.last_bbox_h[track_id] = bbox_h
            self._cleanup(t)
            return self.display_speed[track_id]

        self.last_pixel[track_id]    = (cx_px, cy_px)
        self.last_bbox_h[track_id]   = bbox_h

        raw_x, raw_z = self._sample_point(cx_px, cy_px)
        sx, sz, vx, vz = self.trackers[track_id].update(raw_x, raw_z, t)

        self.history[track_id].append((t, sx, sz))
        while self.history[track_id] and t - self.history[track_id][0][0] > self.window_size:
            self.history[track_id].popleft()

        hist = list(self.history[track_id])
        if len(hist) < MIN_OBS:
            self.last_time[track_id] = t
            self._cleanup(t)
            return self.display_speed[track_id]

        raw_speeds = []
        for i in range(1, len(hist)):
            dt_i = hist[i][0] - hist[i-1][0]
            if dt_i <= 0:
                continue
            dx = hist[i][1] - hist[i-1][1]
            dz = hist[i][2] - hist[i-1][2]
            sp = np.sqrt(dx*dx + dz*dz) / dt_i
            if sp < ABS_MAX_SPEED:
                raw_speeds.append(sp)

        if not raw_speeds:
            self.last_time[track_id] = t
            self._cleanup(t)
            return self.display_speed[track_id]

        speeds_arr = np.array(raw_speeds)
        q1, q3 = np.percentile(speeds_arr, [25, 75])
        iqr = q3 - q1
        fence = q3 + 1.5 * iqr
        clean = speeds_arr[speeds_arr <= fence]
        median_speed = float(np.median(clean)) if len(clean) > 0 else float(np.median(speeds_arr))

        prev = self.current_speed[track_id]
        self.current_speed[track_id] = prev + self.smooth_alpha * (median_speed - prev)

        if t - self.last_display_time[track_id] >= self.display_interval:
            kmh = self.current_speed[track_id] * 3.6
            kmh_rounded = round(kmh * 2) / 2.0
            self.display_speed[track_id] = kmh_rounded / 3.6
            self.last_display_time[track_id] = t

        self.last_time[track_id] = t
        self._cleanup(t)
        return self.display_speed[track_id]

    def _cleanup(self, now):
        dead = [tid for tid, last in self.last_time.items() if now - last > self.timeout]
        for tid in dead:
            self.trackers.pop(tid, None)
            self.current_speed.pop(tid, None)
            self.display_speed.pop(tid, None)
            self.last_display_time.pop(tid, None)
            self.history.pop(tid, None)
            self.last_time.pop(tid, None)
            self.last_pixel.pop(tid, None)
            self.last_bbox_h.pop(tid, None)


# ------------------------------------------------------------------ #
#  Drawing                                                            #
# ------------------------------------------------------------------ #
def _speed_color(kmh):
    ratio = min(kmh / 15.0, 1.0)
    return (0, int(255*(1-ratio)), int(255*ratio))

def draw_bbox(img, bbox, track_id, speed_mps):
    x1, y1, x2, y2 = map(int, bbox)
    kmh   = speed_mps * 3.6
    color = _speed_color(kmh)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    label = f"ID:{track_id}  {kmh:.1f} km/h"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y2+2), (x1+tw+4, y2+th+10), color, -1)
    cv2.putText(img, label, (x1+2, y2+th+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

def draw_topdown_panel(active_data, panel_w=320, panel_h=720,
                       max_depth_z=80.0, min_x=-30.0, max_x=30.0):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (18, 18, 28)
    cv2.putText(panel, "2D TOP-DOWN VIEW", (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 255), 1, cv2.LINE_AA)
    cam_x_px = int((0 - min_x) / (max_x - min_x) * panel_w)
    cam_y_px = panel_h
    for dist in range(5, int(max_depth_z) + 1, 5):
        r_px = int((dist / max_depth_z) * panel_h)
        cv2.circle(panel, (cam_x_px, cam_y_px), r_px, (40, 40, 55), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{dist}m", (cam_x_px + 5, cam_y_px - r_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 100), 1, cv2.LINE_AA)
    for tid, data in active_data.items():
        history = data['history']
        speed = data['speed']
        kmh = speed * 3.6
        color = _speed_color(kmh)
        pts = []
        for t_val, x, z in history:
            px = int((x - min_x) / (max_x - min_x) * panel_w)
            py = int(panel_h - (z / max_depth_z) * panel_h)
            pts.append((px, py))
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(panel, [pts_arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        if pts:
            curr_x, curr_y = pts[-1]
            cv2.circle(panel, (curr_x, curr_y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(panel, f"ID:{tid}", (curr_x + 8, curr_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 255), 1, cv2.LINE_AA)
    return panel

def compose_frame(cam_frame, active_data, out_w, out_h, panel_w=320):
    cam_resized = cv2.resize(cam_frame, (out_w, out_h))
    panel       = draw_topdown_panel(active_data, panel_w=panel_w, panel_h=out_h)
    return np.hstack([cam_resized, panel])


# ------------------------------------------------------------------ #
#  Argument parsing                                                    #
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description='Speed estimation with BoTSORT + YOLO')
    p.add_argument('--empty_frame', required=True)
    p.add_argument('--video',       required=True)
    p.add_argument('--camera_id',   required=True)
    p.add_argument('--det_model',   default='yolov8n_best.onnx')
    # ИЗМЕНЕНО: По умолчанию 0.45 (важно для отсечения фантомов)
    p.add_argument('--conf',        type=float, default=0.45)
    p.add_argument('--device',      default='cuda')
    p.add_argument('--output',      default='video_with_bbox_and_speed.mp4')
    p.add_argument('--out_width',   type=int, default=1280)
    p.add_argument('--out_height',  type=int, default=720)
    p.add_argument('--panel_width', type=int, default=640)
    p.add_argument('--clahe',       type=float, default=0.0)
    p.add_argument('--denoise',     type=float, default=0.0)
    p.add_argument('--sharpen',     type=float, default=0.0)
    p.add_argument('--run_depth',   action='store_true')
    p.add_argument('--no_calibrate', action='store_true')
    p.add_argument('--max_pix_jump',   type=float, default=MAX_PIX_JUMP)
    p.add_argument('--window_sec',     type=float, default=WINDOW_SEC)
    p.add_argument('--ema_alpha',      type=float, default=EMA_ALPHA)
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    args = parse_args()

    global MAX_PIX_JUMP, WINDOW_SEC, EMA_ALPHA
    MAX_PIX_JUMP = args.max_pix_jump
    WINDOW_SEC   = args.window_sec
    EMA_ALPHA    = args.ema_alpha

    cfg_dir = Path('cfg') / args.camera_id
    if args.run_depth:
        run_depth_estimator(
            args.empty_frame, args.camera_id, args.device,
            clahe=args.clahe, denoise=args.denoise, sharpen=args.sharpen,
        )
    elif not (cfg_dir / 'points_map.npy').exists():
        print(f"[ERROR] Depth map not found at {cfg_dir}. "
              f"Run depth_estimator.py first, or pass --run_depth.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {width}×{height} @ {fps:.2f} fps — {total} frames")

    out_total_w = args.out_width + args.panel_width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_total_w, args.out_height))
    if not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for: {args.output}")
        sys.exit(1)

    cuda_ok = 'CUDAExecutionProvider' in ort.get_available_providers()
    
    device_onnx = args.device if (args.device == 'cuda' and cuda_ok) else 'cpu'
    device_tracker = 'cuda:0' if device_onnx == 'cuda' else 'cpu'
    
    if args.device == 'cuda' and not cuda_ok:
        print("[WARNING] CUDA not available — using CPU")

    detector  = YOLOv8Detector(args.det_model, conf_thres=args.conf, device=device_onnx)
    
    tracker   = BoTSORTTracker(frame_rate=fps, device=device_tracker)
    
    speed_est = SpeedEstimator(
        fps=fps,
        window_size=args.window_sec,
        smooth_alpha=args.ema_alpha,
        display_interval=DISPLAY_INTERVAL,
    )
    speed_est.load_depth_map(args.camera_id, video_w=width, video_h=height,
                             apply_calibration=not args.no_calibrate)

    frame_id = 0
    t0       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id  += 1
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        frame   = apply_filters(frame, args.clahe, args.denoise, args.sharpen)
        dets    = detector.detect(frame)
        
        targets = tracker.update(dets, frame)

        active_data = {}
        for t in targets:
            bbox = [t.tlwh[0], t.tlwh[1],
                    t.tlwh[0] + t.tlwh[2],
                    t.tlwh[1] + t.tlwh[3]]
            speed = speed_est.update(t.track_id, bbox, video_time)
            hist  = list(speed_est.history.get(t.track_id, []))
            active_data[t.track_id] = {'speed': speed, 'history': hist}
            draw_bbox(frame, bbox, t.track_id, speed)

        composed = compose_frame(frame, active_data,
                                 args.out_width, args.out_height, args.panel_width)
        writer.write(composed)

        if frame_id % 50 == 0 or frame_id == 1:
            elapsed  = time.time() - t0
            pct      = (frame_id / total * 100) if total > 0 else 0
            spd_fps  = frame_id / elapsed if elapsed > 0 else 0
            eta      = (total - frame_id) / spd_fps if spd_fps > 0 and total > 0 else 0
            print(f"  {frame_id}/{total}  ({pct:.1f}%)  "
                  f"{spd_fps:.1f} fps  ETA {eta:.0f}s", end='\r')

    cap.release()
    writer.release()
    print(f"\n[done] Saved → {args.output}")

if __name__ == '__main__':
    main()
