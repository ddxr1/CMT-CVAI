from ultralytics import YOLO

# YOLOv8 model load
model = YOLO("yolov8n.pt")

# 데이터셋으로 학습 (data.yaml)
# model.train(data="data.yaml", epochs=10, imgsz=640)

# 데이터셋이 너무 커서 사이즈 줄여서 테스트
# model.train(
#     data="data.yaml",
#     epochs=10,
#     imgsz=640,
#     batch=8,
#     fraction=0.3
# )

# 모델 성능 향상을 위한 증강 및 hyperparameter tuning
model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    argument=True,
    fraction=0.5
)