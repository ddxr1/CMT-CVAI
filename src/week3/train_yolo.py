"""
YOLOv8 모델 학습 스크립트
화재 탐지를 위한 YOLO 모델 학습 및 하이퍼파라미터 튜닝
"""

from ultralytics import YOLO
from pathlib import Path
import logging
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_yolo_model(
    model_path: str = "yolov8n.pt",
    data_path: str = "data.yaml",
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    augment: bool = True,
    fraction: float = 0.5,
    device: str = "cpu",
    project: str = "runs/detect",
    name: str = "train"
) -> str:
    """
    YOLOv8 모델 학습
    
    Args:
        model_path: 사전 학습된 모델 경로
        data_path: 데이터셋 설정 파일 경로 (YAML)
        epochs: 학습 에포크 수
        imgsz: 입력 이미지 크기
        batch: 배치 크기
        augment: 데이터 증강 사용 여부
        fraction: 데이터셋 사용 비율 (1.0 = 전체 사용)
        device: 사용할 디바이스 ("cpu", "cuda", "0", "1" 등)
        project: 프로젝트 디렉토리
        name: 실행 이름
        
    Returns:
        학습된 모델의 가중치 파일 경로
    """
    try:
        # 모델 파일 검증
        model_file = Path(model_path)
        if not model_file.exists():
            logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
            logger.info("Ultralytics에서 자동 다운로드를 시도합니다...")
        
        # 데이터셋 파일 검증
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"데이터셋 설정 파일을 찾을 수 없습니다: {data_path}")
        
        # YOLO 모델 로드
        logger.info(f"모델 로딩: {model_path}")
        model = YOLO(model_path)
        
        # 학습 파라미터 출력
        logger.info("=" * 50)
        logger.info("학습 파라미터:")
        logger.info(f"  모델: {model_path}")
        logger.info(f"  데이터셋: {data_path}")
        logger.info(f"  에포크: {epochs}")
        logger.info(f"  이미지 크기: {imgsz}")
        logger.info(f"  배치 크기: {batch}")
        logger.info(f"  데이터 증강: {augment}")
        logger.info(f"  데이터 비율: {fraction}")
        logger.info(f"  디바이스: {device}")
        logger.info("=" * 50)
        
        # 모델 학습
        logger.info("학습 시작...")
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            augment=augment,
            fraction=fraction,
            device=device,
            project=project,
            name=name,
            save=True,
            plots=True
        )
        
        # 학습 완료
        best_model_path = results.save_dir / "weights" / "best.pt"
        logger.info(f"학습 완료! 최적 모델: {best_model_path}")
        
        return str(best_model_path)
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise


def main():
    """메인 함수 - 커맨드라인 인자 처리"""
    parser = argparse.ArgumentParser(description="YOLOv8 모델 학습")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="사전 학습된 모델 경로")
    parser.add_argument("--data", type=str, default="data.yaml", help="데이터셋 설정 파일")
    parser.add_argument("--epochs", type=int, default=20, help="학습 에포크 수")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--augment", action="store_true", default=True, help="데이터 증강 사용")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="데이터 증강 비활성화")
    parser.add_argument("--fraction", type=float, default=0.5, help="데이터셋 사용 비율")
    parser.add_argument("--device", type=str, default="cpu", help="사용할 디바이스")
    parser.add_argument("--project", type=str, default="runs/detect", help="프로젝트 디렉토리")
    parser.add_argument("--name", type=str, default="train", help="실행 이름")
    
    args = parser.parse_args()
    
    try:
        best_model = train_yolo_model(
            model_path=args.model,
            data_path=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            augment=args.augment,
            fraction=args.fraction,
            device=args.device,
            project=args.project,
            name=args.name
        )
        logger.info(f"✅ 학습 완료! 모델 경로: {best_model}")
    except Exception as e:
        logger.error(f"❌ 학습 실패: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # 스크립트로 직접 실행 시 기본 설정으로 학습
    if __name__ == "__main__":
        try:
            best_model = train_yolo_model()
            logger.info(f"✅ 학습 완료! 모델 경로: {best_model}")
        except Exception as e:
            logger.error(f"❌ 학습 실패: {e}")
            exit(1)