import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.45
MODEL_INPUT_SHAPE = (576, 1024)

def load_model(onnx_path):
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def letterbox(img, new_shape=(576, 1024), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess_frame(frame, target_size):
    img, ratio, pad = letterbox(frame, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_norm, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor, ratio, pad

def postprocess(output, orig_shape, input_shape, ratio, pad):
    detections = output[0]
    if detections.shape[0] == 5:
        boxes = detections[:4, :]
        scores = detections[4:5, :]
        person_scores = scores[0, :]
    else:
        boxes = detections[:4, :]
        scores = detections[4:, :]
        person_scores = scores[PERSON_CLASS_ID, :]

    valid_idx = np.where(person_scores > CONFIDENCE_THRESHOLD)[0]

    h_input, w_input = input_shape
    boxes_xyxy = []
    confidences = []

    for idx in valid_idx:
        cx, cy, w, h = boxes[:, idx]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        x1 = max(0, min(x1, w_input))
        y1 = max(0, min(y1, h_input))
        x2 = max(0, min(x2, w_input))
        y2 = max(0, min(y2, h_input))
        boxes_xyxy.append([x1, y1, x2, y2])
        confidences.append(person_scores[idx])

    if not boxes_xyxy:
        return []

    boxes_xywh = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([x1, y1, w, h])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        return []

    r = ratio
    dw, dh = pad
    final_boxes = []
    for i in indices.flatten():
        x1, y1, w, h = boxes_xywh[i]
        x1 -= dw
        y1 -= dh
        x1 /= r
        y1 /= r
        w /= r
        h /= r
        x2 = x1 + w
        y2 = y1 + h
        x1 = max(0, min(x1, orig_shape[1]))
        y1 = max(0, min(y1, orig_shape[0]))
        x2 = max(0, min(x2, orig_shape[1]))
        y2 = max(0, min(y2, orig_shape[0]))
        final_boxes.append([int(x1), int(y1), int(x2), int(y2), confidences[i]])

    return final_boxes

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def main():
    model_path = "models/yolov8s_576x1024_v3.onnx"
    video_path = "data/v003_converted.avi"
    output_video_path = "res/output_video.mp4"


    session, input_name, output_name = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, ratio, pad = preprocess_frame(frame, MODEL_INPUT_SHAPE)
        outputs = session.run([output_name], {input_name: img_tensor})
        boxes = postprocess(outputs[0], frame.shape[:2], MODEL_INPUT_SHAPE, ratio, pad)
        frame = draw_boxes(frame, boxes)

        cv2.imshow('YOLOv8 Person Detection', frame)
        out.write(frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработано кадров: {frame_count}")

if __name__ == "__main__":
    main()
