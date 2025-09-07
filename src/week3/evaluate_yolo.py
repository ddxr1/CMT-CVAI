from ultralytics import YOLO
import matplotlib.pyplot as plt

# 모델 불러오기
model = YOLO("runs/detect/train3/weights/best.pt")

# 성능 평가
metrics = model.val()
print(metrics.results_dict)   # 모든 지표 확인

# 3. Precision / Recall / mAP 시각화
precision = metrics.results_dict["metrics/precision(B)"]
recall = metrics.results_dict["metrics/recall(B)"]
map50 = metrics.results_dict["metrics/mAP50(B)"]
map5095 = metrics.results_dict["metrics/mAP50-95(B)"]

plt.bar(["Precision", "Recall", "mAP50", "mAP50-95"], [precision, recall, map50, map5095])
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("YOLOv8 Model Performance (Validation)")
plt.show()
