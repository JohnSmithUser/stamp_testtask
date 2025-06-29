from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Экспорт модели YOLOv8 в ONNX")
parser.add_argument("--model", type=str, default="model/yolov8n.pt", help="Путь к PyTorch модели YOLOv8n")
args = parser.parse_args()

# Загрузить предобученную модель YOLOv8n
model = YOLO(args.model)

# Экспорт модели в формат ONNX
model.export(format="onnx")