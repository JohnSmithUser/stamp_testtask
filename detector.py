"""Главный файл для запуска детектора транспортных средств с использованием YOLOv8n ONNX."""

from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import yaml
from barrier_state import BarrierStateMachine
import logging
from utils import preprocess, postprocess, draw_ui, box_intersects_roi


# Настройка логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_roi_from_config(path):
    """Загрузка ROI (Region of Interest) из конфигурационного файла.

    Args:
        path (str): Путь к YAML файлу с конфигурацией.

    Returns:
        np.ndarray: Массив координат полигона ROI в формате int32.
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return np.array(config['ROI'], dtype=np.int32), config["WHITE_LIST"]
    except Exception as e:
        logger.error(f"Ошибка при загрузке ROI: {e}")
        raise
        

def run(video_path, model_path, config_path="config.yaml"):
    if not Path(video_path).exists() or not Path(model_path).exists():
        logger.error("Файл видео или модели не найден.")
        return
    
    roi, white_list = load_roi_from_config(config_path)
    
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели ONNX: {e}")
        return
    
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видеофайл.")
        return
    
    state_machine = BarrierStateMachine()
    
    logger.info("Запуск детектора транспортных средств...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Нет кадра. Завершаю работу...")
            break

        img, scale, px, py = preprocess(frame)
        outputs = session.run(None, {input_name: img})
        detections = postprocess(outputs, scale, px, py)
        
        vehicle_in_roi = any(
            box_intersects_roi((x1, y1, x2, y2), roi) and cls_id in white_list
            for x1, y1, x2, y2, conf, cls_id in detections
        )
        
        state = state_machine.update(vehicle_in_roi)
        draw_ui(frame, detections, roi, state, white_list)
        
        cv2.imshow("YOLOv8n ONNX", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()


# === Main ===
if __name__ == "__main__":
    run(
        video_path="cvtest.avi",  # Путь к видеофайлу
        model_path="model/yolov8n.onnx",  # Путь к ONNX модели
        config_path="config.yaml"  # Путь к конфигурационному файлу с ROI
    )
