"""
이미지 전처리 모듈
데이터셋 이미지를 필터링하고 전처리하는 유틸리티
"""

import cv2
import numpy as np
from datasets import load_dataset
from pathlib import Path
from typing import Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상수 정의
DARK_MEAN_THRESH = 40      # 평균 밝기 임계값(0~255). 이보다 어두우면 제거
MIN_AREA_RATIO = 0.05      # 가장 큰 객체 면적 / 전체 면적 비율이 이보다 작으면 제거
TARGET_SIZE = (224, 224)   # 리사이즈 크기

def is_too_dark(gray: np.ndarray, mean_thresh: float = DARK_MEAN_THRESH) -> bool:
    """
    이미지가 너무 어두운지 확인
    
    Args:
        gray: 그레이스케일 이미지
        mean_thresh: 평균 밝기 임계값
        
    Returns:
        너무 어두우면 True
    """
    if gray is None or gray.size == 0:
        return True
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


def preprocess_image(img: np.ndarray, 
                    target_size: Tuple[int, int] = TARGET_SIZE,
                    apply_flip: bool = True) -> np.ndarray:
    """
    이미지 전처리 파이프라인
    
    Args:
        img: 원본 BGR 이미지
        target_size: 리사이즈 목표 크기
        apply_flip: 좌우 반전 적용 여부 (데이터 증강)
        
    Returns:
        전처리된 이미지 (0~1 범위 정규화)
    """
    if img is None:
        raise ValueError("입력 이미지가 None입니다")
    
    # 1) 크기 조정
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # 2) Grayscale & Normalize (0~1)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0  # 정규화

    # 3) 노이즈 제거 (Gaussian Blur)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    # 4) 데이터 증강 (좌우 반전)
    if apply_flip:
        result = cv2.flip(blurred, 1)
    else:
        result = blurred

    return result

def process_dataset(dataset_name: str = "food101",
                   split: str = "train[:5]",
                   save_dir: Optional[str] = None,
                   skip_dark: bool = True,
                   skip_small: bool = True) -> int:
    """
    데이터셋 이미지 전처리 및 필터링
    
    Args:
        dataset_name: 데이터셋 이름
        split: 데이터셋 분할
        save_dir: 저장 디렉토리 경로 (None이면 프로젝트 루트 사용)
        skip_dark: 어두운 이미지 건너뛰기 여부
        skip_small: 작은 객체 이미지 건너뛰기 여부
        
    Returns:
        처리된 이미지 개수
    """
    try:
        # 저장 디렉토리 설정
        if save_dir is None:
            project_root = Path(__file__).parent.parent.parent
            save_dir = project_root / "preprocessed_samples"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"저장 디렉토리: {save_dir}")
        
        # 데이터셋 로드
        logger.info(f"데이터셋 로딩: {dataset_name} ({split})")
        dataset = load_dataset(dataset_name, split=split)
        
        kept = 0
        total = len(dataset)
        
        for i, sample in enumerate(dataset):
            try:
                img = sample["image"]
                img_np = np.array(img)  # RGB
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # 이상치 필터링
                if skip_dark and is_too_dark(gray):
                    logger.debug(f"[skip] {i}/{total}: too dark (mean={gray.mean():.1f})")
                    continue

                if skip_small and is_object_too_small(gray):
                    logger.debug(f"[skip] {i}/{total}: object too small")
                    continue

                # 전처리
                processed = preprocess_image(img_bgr)
                save_path = save_dir / f"sample_{i}.jpg"

                # 0~1 float → uint8 저장
                if cv2.imwrite(str(save_path), (processed * 255).astype(np.uint8)):
                    kept += 1
                    if kept % 10 == 0:  # 진행 상황 로그
                        logger.info(f"진행: {kept}/{total} 처리 완료")
                else:
                    logger.warning(f"이미지 저장 실패: {save_path}")
                    
            except Exception as e:
                logger.error(f"샘플 {i} 처리 중 오류: {e}")
                continue
        
        logger.info(f"[완료] 총 {kept}/{total}개 이미지 처리")
        return kept
        
    except Exception as e:
        logger.error(f"데이터셋 처리 중 오류 발생: {e}")
        return 0


if __name__ == "__main__":
    # 메인 실행
    process_dataset()