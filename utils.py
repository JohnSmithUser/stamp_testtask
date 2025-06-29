import cv2
import numpy as np

def letterbox(img, new_size=640):
    """Изменение размера изображения с сохранением соотношения сторон и добавлением отступов.

    Args:
        img (np.ndarray): Исходное изображение.
        new_size (int): Новый размер для изменения размера изображения.

    Returns:
        tuple: Измененное изображение с отступами, масштаб, отступ по X и отступ по Y.
    """
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
    """Предобработка кадра для подачи в модель YOLOv8n.

    Args:
        frame (np.ndarray): Кадр изображения.

    Returns:
        tuple: Измененное изображение, масштаб, отступ по X и отступ по Y.
    """
    img, scale, px, py = letterbox(frame)
    img = img.transpose(2, 0, 1)[None] / 255.0  # BCHW
    return img.astype(np.float32), scale, px, py


def postprocess(output, scale, px, py, conf_th=0.25, iou_th=0.45):
    """Постобработка выходных данных модели YOLOv8n.

    Args:
        output (list): Выходные данные модели YOLOv8n.
        scale (float): Масштаб, использованный при изменении размера изображения.
        px (int): Отступ по X, добавленный при изменении размера изображения.
        py (int): Отступ по Y, добавленный при изменении размера изображения.
        conf_th (float, optional): Порог уверенности для фильтрации предсказаний. Defaults to 0.25.
        iou_th (float, optional): Порог IoU для NMS. Defaults to 0.45.

    Returns:
        list: Список обнаруженных объектов в формате (x1, y1, x2, y2, confidence, class_id).
    """
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
    """Проверка пересечения ограничивающего прямоугольника с ROI.

    Args:
        box (tuple): Координаты ограничивающего прямоугольника в формате (x1, y1, x2, y2).
        roi_polygon (np.ndarray): Массив координат полигона ROI.

    Returns:
        bool: True, если ограничивающий прямоугольник пересекает ROI, иначе False.
    """
    x1, y1, x2, y2 = box
    box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    inter_area, _ = cv2.intersectConvexConvex(roi_polygon.astype(np.float32), box_poly.astype(np.float32))
    return inter_area > 0


def draw_ui(frame, detections, roi, state, white_list):
    """Отрисовка пользовательского интерфейса на кадре.

    Args:
        frame (np.ndarray): Кадр изображения, на котором нужно отрисовать UI.
        detections (list): Список обнаруженных объектов в формате (x1, y1, x2, y2, confidence, class_id).
        roi (np.ndarray): Массив координат полигона ROI.
        state (str): Текущее состояние машины состояний (например, "VEHICLE_WAITING").
        white_list (dict): Словарь с ID классов и их метками для отрисовки.
    """
    for x1, y1, x2, y2, conf, cls_id in detections:
        label = white_list.get(cls_id)
        if label:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.polylines(frame, [roi], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, f"State: {state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if state == "VEHICLE_WAITING":
        cv2.putText(frame, "VEHICLE DETECTED", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)