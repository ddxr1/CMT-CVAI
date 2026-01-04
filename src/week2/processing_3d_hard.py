"""
심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
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
        colormap: OpenCV 컬러맵 타입
        
    Returns:
        (그레이스케일 이미지, 컬러맵 깊이 이미지) 튜플
        
    Raises:
        ValueError: 이미지가 None이거나 유효하지 않은 경우
    """
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    
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


def generate_point_cloud(gray: np.ndarray,
                        normalize_z: bool = True) -> np.ndarray:
    """
    2D 그레이스케일 이미지를 3D 포인트 클라우드로 변환
    
    Args:
        gray: 그레이스케일 이미지 (H x W)
        normalize_z: Z 값을 0~1 범위로 정규화할지 여부
        
    Returns:
        3D 포인트 클라우드 배열 (H x W x 3), 각 포인트는 (X, Y, Z) 좌표
        
    Raises:
        ValueError: 입력 이미지가 None이거나 그레이스케일이 아닌 경우
    """
    if gray is None:
        raise ValueError("입력된 GrayScale 이미지가 없습니다.")
    
    if len(gray.shape) != 2:
        raise ValueError("Grayscale 이미지가 아닙니다. 2차원 배열이어야 합니다.")
    
    # 이미지 크기
    h, w = gray.shape[:2]
    
    # X, Y 좌표 그리드 생성
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Z 값을 깊이로 사용 (정규화 옵션)
    Z = gray.astype(np.float32)
    if normalize_z:
        Z = Z / 255.0  # 0~1 범위로 정규화
    
    # 3D 좌표 스택 (H x W x 3)
    points_3d = np.dstack((X, Y, Z))
    
    return points_3d


def process_3d_point_cloud(input_path: str,
                          depth_output_path: Optional[str] = None,
                          show_preview: bool = False) -> Optional[np.ndarray]:
    """
    이미지 파일에서 3D 포인트 클라우드 생성
    
    Args:
        input_path: 입력 이미지 경로
        depth_output_path: 깊이 맵 저장 경로 (None이면 저장하지 않음)
        show_preview: 미리보기 창 표시 여부
        
    Returns:
        3D 포인트 클라우드 배열 또는 None (실패 시)
    """
    try:
        # 입력 파일 검증
        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
            return None
        
        # 이미지 로드
        img = cv2.imread(str(input_file))
        if img is None:
            logger.error(f"이미지를 로드할 수 없습니다: {input_path}")
            return None
        
        # 깊이 맵 생성
        gray, depth_map = generate_depth_map(img)
        
        # 3D 포인트 클라우드 생성
        points_3d = generate_point_cloud(gray)
        
        logger.info(f"3D 포인트 클라우드 생성 완료: {points_3d.shape}")
        
        # 깊이 맵 저장
        if depth_output_path:
            output_file = Path(depth_output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(output_file), depth_map):
                logger.info(f"깊이 맵 저장 완료: {depth_output_path}")
            else:
                logger.warning(f"깊이 맵 저장 실패: {depth_output_path}")
        
        # 미리보기 표시
        if show_preview:
            cv2.imshow('Depth Map', depth_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return points_3d
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    # 프로젝트 루트 기준 경로 설정
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "inputs" / "sample.jpg"
    depth_output_path = project_root / "outputs" / "depth_map2.jpg"
    
    # 처리 실행
    points_3d = process_3d_point_cloud(
        str(input_path),
        depth_output_path=str(depth_output_path),
        show_preview=True
    )
    
    if points_3d is not None:
        logger.info(f"포인트 클라우드 통계: shape={points_3d.shape}, "
                   f"Z 범위=[{points_3d[:, :, 2].min():.3f}, {points_3d[:, :, 2].max():.3f}]")
    else:
        print("처리 실패")