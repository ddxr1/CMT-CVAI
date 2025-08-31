import cv2
import numpy as np
from datasets import load_dataset
import os


DARK_MEAN_THRESH = 40      # 평균 밝기 임계값(0~255). 이보다 어두우면 제거
MIN_AREA_RATIO = 0.05      # 가장 큰 객체 면적 / 전체 면적 비율이 이보다 작으면 제거
TARGET_SIZE = (224, 224)   # 리사이즈 크기


dataset = load_dataset("food101", split="train[:5]")
save_dir = "preprocessed_samples"   
os.makedirs(save_dir, exist_ok=True)

def is_too_dark(gray: np.ndarray, mean_thresh: float = DARK_MEAN_THRESH) -> bool:
    return float(gray.mean()) < mean_thresh

def is_object_too_small(gray: np.ndarray, min_area_ratio: float = MIN_AREA_RATIO) -> bool:
    """
    최대 연결영역을 '주요 객체'로 보고, 그 면적이 전체 대비 너무 작으면 True.
    라벨이 없다고 가정하고 이진화+모폴로지+컨투어로 근사 판단.
    """
    h, w = gray.shape[:2]
    img_area = float(h * w)

    # 노이즈 완화 & Otsu 이진화
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 전경이 검정/흰색이 뒤집힌 경우 보정
    if (th == 255).mean() < 0.5:
        th = 255 - th

    # 작은 구멍 메우기
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 컨투어 탐색
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return True  # 전경이 거의 없음 → 너무 작다고 판단

    max_area = max(cv2.contourArea(c) for c in cnts)
    area_ratio = max_area / img_area
    return area_ratio < min_area_ratio
# -------------------------------------------

def preprocess_image(img):
    # 1) 크기 조정
    img = cv2.resize(img, TARGET_SIZE)

    # 2) Grayscale & Normalize (0~1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0  # <- 225가 아니라 255가 맞음!

    # 3) 노이즈 제거 (Gaussian Blur)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)

    # 4) 데이터 증강 (좌우 반전)
    flip = cv2.flip(blur, 1)

    return flip

kept = 0
for i, sample in enumerate(dataset):
    img = sample["image"]
    img_np = np.array(img)                # RGB
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --------- 심화: 이상치 필터링 ---------
    if is_too_dark(gray):
        print(f"[skip] {i}: too dark (mean={gray.mean():.1f})")
        continue

    if is_object_too_small(gray):
        print(f"[skip] {i}: object too small")
        continue
    # --------------------------------------

    processed = preprocess_image(img_bgr)
    save_path = os.path.join(save_dir, f"sample_{i}.jpg")

    # 0~1 float → uint8 저장
    cv2.imwrite(save_path, (processed * 255).astype(np.uint8))
    print(f"Saved: {save_path}")
    kept += 1

print(f"[DONE] kept={kept}")