"""
YOLOv8 객체 탐지 스크립트
학습된 YOLO 모델을 사용하여 이미지에서 화재 탐지
"""

import cv2
import glob
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_objects_in_image(
    model: YOLO,
    image: cv2.typing.MatLike,
    confidence_threshold: float = 0.05
) -> List[Tuple[int, int, int, int, str, float]]:
    """
    이미지에서 객체 탐지
    
    Args:
        model: YOLO 모델
        image: 입력 이미지 (BGR)
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        탐지 결과 리스트 [(x1, y1, x2, y2, label, confidence), ...]
    """
    if image is None:
        return []
    
    # 객체 탐지 실행
    results = model(image, conf=confidence_threshold, verbose=False)
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                detections.append((x1, y1, x2, y2, label, conf))
    
    return detections


def visualize_detections(
    image: cv2.typing.MatLike,
    detections: List[Tuple[int, int, int, int, str, float]],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> cv2.typing.MatLike:
    """
    탐지 결과를 이미지에 시각화
    
    Args:
        image: 원본 이미지
        detections: 탐지 결과 리스트
        color: 바운딩 박스 색상 (BGR)
        
    Returns:
        시각화된 이미지
    """
    result_image = image.copy()
    
    for x1, y1, x2, y2, label, conf in detections:
        # 바운딩 박스 그리기
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # 라벨과 신뢰도 표시
        label_text = f"{label} {conf:.2f}"
        cv2.putText(result_image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result_image


def process_images(
    model_path: str,
    input_dir: str,
    output_dir: str,
    pattern: str = "*.jpg",
    confidence_threshold: float = 0.05,
    show_preview: bool = False
) -> int:
    """
    디렉토리의 이미지들을 배치 처리하여 객체 탐지
    
    Args:
        model_path: YOLO 모델 경로
        input_dir: 입력 이미지 디렉토리
        output_dir: 출력 이미지 디렉토리
        pattern: 파일 패턴 (예: "fire_*.jpg")
        confidence_threshold: 신뢰도 임계값
        show_preview: 미리보기 창 표시 여부
        
    Returns:
        처리된 이미지 개수
    """
    try:
        # 모델 로드
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        logger.info(f"모델 로딩: {model_path}")
        model = YOLO(model_path)
        
        # 디렉토리 설정
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 목록
        image_pattern = str(input_path / pattern)
        image_paths = glob.glob(image_pattern)
        
        if not image_paths:
            logger.warning(f"입력 디렉토리에서 이미지를 찾을 수 없습니다: {image_pattern}")
            return 0
        
        logger.info(f"처리할 이미지: {len(image_paths)}개")
        
        processed_count = 0
        
        for image_path in image_paths:
            try:
                # 이미지 로드
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"이미지 로드 실패: {image_path}")
                    continue
                
                # 객체 탐지
                detections = detect_objects_in_image(model, img, confidence_threshold)
                
                if detections:
                    logger.info(f"{Path(image_path).name}: {len(detections)}개 객체 탐지")
                
                # 결과 시각화
                result_img = visualize_detections(img, detections)
                
                # 결과 저장
                filename = Path(image_path).name
                save_path = output_path / f"detect_{filename}"
                if cv2.imwrite(str(save_path), result_img):
                    processed_count += 1
                else:
                    logger.warning(f"이미지 저장 실패: {save_path}")
                
                # 미리보기 표시
                if show_preview:
                    cv2.imshow("YOLOv8 Detection", result_img)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                    
            except Exception as e:
                logger.error(f"이미지 처리 중 오류 ({image_path}): {e}")
                continue
        
        if show_preview:
            cv2.destroyAllWindows()
        
        logger.info(f"처리 완료: {processed_count}/{len(image_paths)}개 이미지")
        return processed_count
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        return 0


def main():
    """메인 함수 - 커맨드라인 인자 처리"""
    parser = argparse.ArgumentParser(description="YOLOv8 객체 탐지")
    parser.add_argument("--model", type=str, default="runs/detect/train3/weights/best.pt",
                       help="YOLO 모델 경로")
    parser.add_argument("--input", type=str, default="../../inputs/yolo_detection_samples/",
                       help="입력 이미지 디렉토리")
    parser.add_argument("--output", type=str, default="../../outputs/fire_detection/",
                       help="출력 이미지 디렉토리")
    parser.add_argument("--pattern", type=str, default="fire_*.jpg",
                       help="파일 패턴")
    parser.add_argument("--conf", type=float, default=0.05,
                       help="신뢰도 임계값")
    parser.add_argument("--show", action="store_true",
                       help="미리보기 창 표시")
    
    args = parser.parse_args()
    
    processed = process_images(
        model_path=args.model,
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        confidence_threshold=args.conf,
        show_preview=args.show
    )
    
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    # 스크립트로 직접 실행 시 기본 설정으로 처리
    project_root = Path(__file__).parent.parent.parent
    
    processed = process_images(
        model_path="runs/detect/train3/weights/best.pt",
        input_dir=str(project_root / "inputs" / "yolo_detection_samples"),
        output_dir=str(project_root / "outputs" / "fire_detection"),
        pattern="fire_*.jpg",
        confidence_threshold=0.05,
        show_preview=False
    )