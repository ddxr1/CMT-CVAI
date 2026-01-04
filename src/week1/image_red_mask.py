"""
빨간색 마스킹 모듈
이미지에서 빨간색 영역을 추출하는 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상수 정의
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])


def create_red_mask(hsv_image: np.ndarray, 
                    lower_red1: np.ndarray = LOWER_RED1,
                    upper_red1: np.ndarray = UPPER_RED1,
                    lower_red2: np.ndarray = LOWER_RED2,
                    upper_red2: np.ndarray = UPPER_RED2) -> np.ndarray:
    """
    HSV 이미지에서 빨간색 영역 마스크 생성
    
    Args:
        hsv_image: HSV 색공간 이미지
        lower_red1: 첫 번째 빨간색 범위 하한값
        upper_red1: 첫 번째 빨간색 범위 상한값
        lower_red2: 두 번째 빨간색 범위 하한값 (색상이 180도를 넘어가는 경우)
        upper_red2: 두 번째 빨간색 범위 상한값
        
    Returns:
        빨간색 영역 마스크 (uint8)
    """
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # 더 명확한 논리 연산 사용
    return mask


def extract_red_regions(image: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    원본 이미지에서 빨간색 영역만 추출
    
    Args:
        image: 원본 BGR 이미지
        mask: 빨간색 마스크 (None이면 자동 생성)
        
    Returns:
        빨간색 영역만 추출된 이미지
    """
    if mask is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = create_red_mask(hsv)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def process_red_mask(input_path: str, 
                    output_path: str,
                    show_preview: bool = False) -> bool:
    """
    이미지 파일에서 빨간색 영역 추출 및 저장
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        show_preview: 미리보기 창 표시 여부
        
    Returns:
        처리 성공 여부
    """
    try:
        # 입력 파일 검증
        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
            return False
        
        # 이미지 로드
        img = cv2.imread(str(input_file))
        if img is None:
            logger.error(f"이미지를 로드할 수 없습니다: {input_path}")
            return False
        
        # HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 빨간색 마스크 생성
        mask = create_red_mask(hsv)
        
        # 빨간색 영역 추출
        result = extract_red_regions(img, mask)
        
        # 출력 디렉토리 생성
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장
        if not cv2.imwrite(str(output_file), result):
            logger.error(f"이미지 저장 실패: {output_path}")
            return False
        
        logger.info(f"처리 완료: {output_path}")
        
        # 미리보기 표시
        if show_preview:
            cv2.imshow('원본', img)
            cv2.imshow('빨간색 필터', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        return False


if __name__ == "__main__":
    # 프로젝트 루트 기준 경로 설정
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "inputs" / "sample.jpg"
    output_path = project_root / "outputs" / "sample_output.jpg"
    
    # 처리 실행
    success = process_red_mask(str(input_path), str(output_path), show_preview=True)
    if not success:
        print("처리 실패")