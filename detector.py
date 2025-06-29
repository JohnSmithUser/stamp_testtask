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
        config (dict): Словарь с конфигурацией.
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    except Exception as e:
        logger.error(f"Ошибка при загрузке ROI: {e}")
        raise


def video_writer(cap, output_path="result/result.avi"):
    """Создание объекта для записи видео.

    Args:
        cap (cv2.VideoCapture): Объект захвата видео.
        output_path (str): Путь к выходному видеофайлу.

    Returns:
        cv2.VideoWriter: Объект для записи видео.
    """
    # Создаём директорию, если не существует
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем размер кадра из видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    logger.info(f"Видео будет сохранено в {output_path}")
    
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        

def run(video_path, model_path, roi, white_list, show: bool, output_path="result/result.avi"):
    """Запуск детектора транспортных средств на видео с использованием модели YOLOv8n в формате ONNX.

    Args:
        video_path (str): Путь к видеофайлу, который нужно обработать.
        model_path (str): Путь к ONNX модели YOLOv8n.
    """
    if not Path(video_path).exists() or not Path(model_path).exists():
        logger.error("Файл видео или модели не найден.")
        return
    
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
    if not show:
        writer = video_writer(cap, output_path)
    
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
        
        if show:
            cv2.imshow("YOLOv8n ONNX", cv2.resize(frame, (1280, 720)))
            if cv2.waitKey(1) == 27:  # ESC key to exit
                break
        else:
            writer.write(frame)
    
    cap.release()
    cv2.destroyAllWindows()


# === Main ===
if __name__ == "__main__":
    config = load_roi_from_config("config.yaml")

    run(
        video_path=config["video_path"],
        model_path=config["model_path"],
        roi=np.array(config['ROI'], dtype=np.int32),
        white_list=config['WHITE_LIST'],
        show=config["show"],
        output_path=config["output_path"]
    )
