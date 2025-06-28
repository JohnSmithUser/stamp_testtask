from ultralytics import YOLO

# Загрузить предобученную модель YOLOv8n
model = YOLO("model/yolov8n.pt")  # можно заменить на свой путь к модели

# Экспорт модели в формат ONNX
model.export(format="onnx")