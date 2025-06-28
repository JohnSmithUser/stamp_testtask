import cv2
import numpy as np
import onnxruntime as ort
import yaml
from barrier_state import BarrierStateMachine


white_list = {
    2: "car",        # ID для автомобиля
    3: "motorcycle", # ID для мотоцикла
    7: "truck",      # ID для грузовика
    5: "bus",        # ID для автобуса
}


def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    pad_y = (new_size - nh) // 2
    pad_x = (new_size - nw) // 2
    padded = cv2.copyMakeBorder(img_resized, pad_y, new_size - nh - pad_y,
                                pad_x, new_size - nw - pad_x,
                                cv2.BORDER_CONSTANT, (114, 114, 114))
    return padded, scale, pad_x, pad_y


def preprocess(frame):
    img, scale, px, py = letterbox(frame)
    img = img.transpose(2, 0, 1)[None] / 255.0  # BCHW
    return img.astype(np.float32), scale, px, py


def postprocess(output, scale, px, py, conf_th=0.25, iou_th=0.45):
    out = output[0]
    if out.shape[1] == 84:
        out = out.squeeze(0).transpose(1, 0)  # (8400, 84)
    else:
        out = out.squeeze(0)

    boxes = out[:, :4]
    scores = out[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_th
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # xywh → xyxy
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    # Undo letterbox padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale

    # Apply NMS
    rects = boxes.tolist()
    idxs = cv2.dnn.NMSBoxes(rects, confidences.tolist(), conf_th, iou_th)
    if len(idxs) == 0:
        return []

    results = []
    for i in idxs.flatten():
        x1, y1, x2, y2 = map(int, rects[i])
        results.append((x1, y1, x2, y2, float(confidences[i]), int(class_ids[i])))
    return results


def box_intersects_roi(box, roi_polygon):
    x1, y1, x2, y2 = box
    box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    inter_area, _ = cv2.intersectConvexConvex(roi_polygon.astype(np.float32), box_poly.astype(np.float32))
    return inter_area > 0


# === Main ===
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    roi = np.array(config['roi_polygon'], dtype=np.int32)

    session = ort.InferenceSession("model/yolov8n.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture("cvtest.avi")
    
    state_machine = BarrierStateMachine()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Нет кадра. Завершаю работу...")
            break

        img, scale, px, py = preprocess(frame)
        outputs = session.run(None, {input_name: img})
        detections = postprocess(outputs, scale, px, py)
        
        vehicle_in_roi = False

        for x1, y1, x2, y2, conf, cls_id in detections:
            if box_intersects_roi((x1, y1, x2, y2), roi) and cls_id in white_list:
                vehicle_in_roi = True
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{white_list[cls_id]}:{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        state = state_machine.update(vehicle_in_roi)
        cv2.polylines(frame, [roi], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, f"State: {state}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        if state == "VEHICLE_WAITING":
            cv2.putText(frame, "VEHICLE DETECTED", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.imshow("YOLOv8n ONNX", cv2.resize(frame, (1280, 720)))
        
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
