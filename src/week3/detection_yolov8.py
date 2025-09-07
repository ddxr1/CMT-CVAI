import cv2
import glob
import os
from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

# 2. 입력/출력 경로 설정
input_dir = "../../inputs/yolo_detection_samples/"
output_dir = "../../outputs/fire_detection/"
os.makedirs(output_dir, exist_ok=True)
image_paths = glob.glob(os.path.join(input_dir, "fire_*.jpg"))
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print("image load failed")
        continue

    # 객체 탐지 실행
    results = model(img, conf=0.05)

    # 탐지된 객체 시각화
    for rst in results:
        for box in rst.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = rst.names[int(box.cls[0])]
            conf = box.conf[0]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"detect_{filename}")
    cv2.imwrite(save_path, img)

    # 결과 출력
    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()