"""
기본적인 Depth Map 생성 모듈
OpenCV를 활용한 깊이 맵 생성 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_depth_map(image: np.ndarray, 
                      colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray]:
    """
    그레이스케일 이미지를 컬러맵으로 변환하여 깊이 맵 생성
    
    Args:
        image: 입력 BGR 이미지
        colormap: OpenCV 컬러맵 타입 (기본값: JET)
        
    Returns:
        (그레이스케일 이미지, 컬러맵 깊이 이미지) 튜플
        
    Raises:
        ValueError: 이미지가 None이거나 유효하지 않은 경우
    """
    if image is None:
        raise ValueError("입력 이미지가 None입니다.")
    
    if not isinstance(image, np.ndarray) or len(image.shape) < 2:
        raise ValueError("유효하지 않은 이미지 형식입니다.")
    
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 컬러맵 적용
    depth_map = cv2.applyColorMap(gray, colormap)
    
    return gray, depth_map


def process_depth_map(input_path: str,
                     output_path: str,
                     colormap: int = cv2.COLORMAP_JET,
                     show_preview: bool = False) -> bool:
    """
    이미지 파일에서 깊이 맵 생성 및 저장
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        colormap: 컬러맵 타입
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
        
        # 깊이 맵 생성
        gray, depth_map = generate_depth_map(img, colormap)
        
        # 출력 디렉토리 생성
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장
        if not cv2.imwrite(str(output_file), depth_map):
            logger.error(f"이미지 저장 실패: {output_path}")
            return False
        
        logger.info(f"처리 완료: {output_path}")
        
        # 미리보기 표시
        if show_preview:
            cv2.imshow("원본", img)
            cv2.imshow("Depth Map", depth_map)
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
    output_path = project_root / "outputs" / "depth_map1.jpg"
    
    # 처리 실행
    success = process_depth_map(str(input_path), str(output_path), show_preview=True)
    if not success:
        print("처리 실패")