import cv2
import yaml
import onnxruntime as ort
import numpy as np

# ==================== Конфигурация ====================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
roi = config['roi']

# ==================== ONNX и препроцесс ====================
session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, 640, 640]


def preprocess(frame, input_size=(640, 640)):
    resized = cv2.resize(frame, input_size)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img


def postprocess(outputs, input_shape, orig_shape, conf_thresh=0.5):
    predictions = outputs[0][0]
    boxes = []
    for pred in predictions:
        conf = pred[4]
        cls_id = int(pred[5])
        if conf > conf_thresh and cls_id in [2, 3]:  # 2: car, 3: motorcycle
            x_center, y_center, w, h = pred[:4]
            x1 = int((x_center - w / 2) * orig_shape[1] / input_shape[2])
            y1 = int((y_center - h / 2) * orig_shape[0] / input_shape[1])
            x2 = int((x_center + w / 2) * orig_shape[1] / input_shape[2])
            y2 = int((y_center + h / 2) * orig_shape[0] / input_shape[1])
            boxes.append(((x1, y1, x2, y2), conf, cls_id))
    return boxes


# ==================== Машина состояний ====================
class BarrierStateMachine:
    def __init__(self, min_frames_presence=5):
        self.state = "NO_VEHICLE"
        self.presence_counter = 0
        self.min_frames_presence = min_frames_presence

    def update(self, vehicle_detected: bool):
        if vehicle_detected:
            self.presence_counter += 1
        else:
            self.presence_counter = 0

        if self.presence_counter >= self.min_frames_presence:
            self.state = "VEHICLE_WAITING"
        elif self.presence_counter > 0:
            self.state = "VEHICLE_APPROACHING"
        else:
            self.state = "NO_VEHICLE"

        return self.state


# ==================== Основной цикл ====================
cap = cv2.VideoCapture("input_video.mp4")
state_machine = BarrierStateMachine()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Создаем blackout кадр
    masked_frame = np.zeros_like(frame)
    roi_frame = frame[roi['y']:roi['y'] + roi['height'],
                      roi['x']:roi['x'] + roi['width']]
    masked_frame[roi['y']:roi['y'] + roi['height'],
                 roi['x']:roi['x'] + roi['width']] = roi_frame

    input_tensor = preprocess(masked_frame)
    outputs = session.run(None, {input_name: input_tensor})
    boxes = postprocess(outputs, input_shape, frame.shape)

    vehicle_in_roi = False
    for (x1, y1, x2, y2), conf, cls_id in boxes:
        label = "Car" if cls_id == 2 else "Motorcycle"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if (x1 < roi['x'] + roi['width'] and x2 > roi['x'] and
            y1 < roi['y'] + roi['height'] and y2 > roi['y']):
            vehicle_in_roi = True

    state = state_machine.update(vehicle_in_roi)

    # Отрисовка ROI и состояния
    cv2.rectangle(frame, (roi['x'], roi['y']),
                  (roi['x'] + roi['width'], roi['y'] + roi['height']),
                  (255, 0, 0), 2)
    cv2.putText(frame, f"State: {state} (blackout ROI)", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if state == "VEHICLE_WAITING":
        cv2.putText(frame, "АВТОМОБИЛЬ ОБНАРУЖЕН", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
