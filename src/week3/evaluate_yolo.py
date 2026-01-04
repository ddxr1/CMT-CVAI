"""
YOLOv8 모델 성능 평가 스크립트
학습된 모델의 Precision, Recall, mAP 등을 평가하고 시각화
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    data_path: Optional[str] = None,
    imgsz: int = 640,
    save_plot: bool = True,
    plot_output_path: Optional[str] = None
) -> dict:
    """
    YOLO 모델 성능 평가
    
    Args:
        model_path: 학습된 YOLO 모델 경로
        data_path: 평가 데이터셋 경로 (YAML), None이면 학습 시 사용한 데이터 사용
        imgsz: 평가 이미지 크기
        save_plot: 플롯 저장 여부
        plot_output_path: 플롯 저장 경로 (None이면 자동 생성)
        
    Returns:
        평가 지표 딕셔너리
    """
    try:
        # 모델 파일 검증
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 모델 로드
        logger.info(f"모델 로딩: {model_path}")
        model = YOLO(model_path)
        
        # 데이터셋 경로 확인
        if data_path:
            data_file = Path(data_path)
            if not data_file.exists():
                logger.warning(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
                data_path = None
        
        # 성능 평가
        logger.info("모델 평가 중...")
        metrics = model.val(data=data_path, imgsz=imgsz, verbose=True)
        
        # 결과 출력
        results_dict = metrics.results_dict
        logger.info("=" * 50)
        logger.info("평가 결과:")
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 50)
        
        # 주요 지표 추출 (버전별로 키 이름이 다를 수 있음)
        precision = results_dict.get("metrics/precision(B)", 0.0)
        recall = results_dict.get("metrics/recall(B)", 0.0)
        map50 = results_dict.get("metrics/mAP50(B)", 0.0)
        map5095 = results_dict.get("metrics/mAP50-95(B)", 0.0)
        
        # 시각화
        if save_plot:
            plt.figure(figsize=(10, 6))
            metrics_list = [precision, recall, map50, map5095]
            labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
            
            bars = plt.bar(labels, metrics_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            plt.ylim(0, 1.0)
            plt.ylabel("Score", fontsize=12)
            plt.title("YOLOv8 Model Performance (Validation)", fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, metrics_list):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # 저장
            if plot_output_path is None:
                plot_output_path = Path(model_path).parent.parent / "evaluation_plot.png"
            else:
                plot_output_path = Path(plot_output_path)
            
            plot_output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(plot_output_path), dpi=300, bbox_inches='tight')
            logger.info(f"플롯 저장: {plot_output_path}")
            
            # 표시
            plt.show()
        
        return results_dict
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        raise


def main():
    """메인 함수 - 커맨드라인 인자 처리"""
    parser = argparse.ArgumentParser(description="YOLOv8 모델 성능 평가")
    parser.add_argument("--model", type=str, default="runs/detect/train3/weights/best.pt",
                       help="YOLO 모델 경로")
    parser.add_argument("--data", type=str, default=None,
                       help="평가 데이터셋 경로 (YAML)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="평가 이미지 크기")
    parser.add_argument("--no-plot", dest="save_plot", action="store_false",
                       help="플롯 저장 비활성화")
    parser.add_argument("--plot-output", type=str, default=None,
                       help="플롯 저장 경로")
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(
            model_path=args.model,
            data_path=args.data,
            imgsz=args.imgsz,
            save_plot=args.save_plot,
            plot_output_path=args.plot_output
        )
        logger.info("✅ 평가 완료!")
        return 0
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}")
        return 1


if __name__ == "__main__":
    # 스크립트로 직접 실행 시 기본 설정으로 평가
    try:
        results = evaluate_model(
            model_path="runs/detect/train3/weights/best.pt",
            save_plot=True
        )
        logger.info("✅ 평가 완료!")
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}")
        exit(1)
