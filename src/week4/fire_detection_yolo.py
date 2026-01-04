"""
YOLOv8 기반 화재 탐지 모듈
사전 학습된 YOLOv8 모델을 사용하여 화재를 탐지합니다.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import os

class FireDetectorYOLO:
    """YOLOv8을 사용한 화재 탐지 클래스"""
    
    def __init__(self, 
                 model_path: str = "../week3/runs/detect/train/weights/best.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        YOLO 화재 탐지기 초기화
        
        Args:
            model_path: 학습된 YOLO 모델 경로
            confidence_threshold: 탐지 신뢰도 임계값
            device: 사용할 디바이스 (cpu/cuda)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 모델 로드
        try:
            # Path 객체로 변환하여 경로 처리 개선
            from pathlib import Path
            
            model_file = Path(model_path)
            if not model_file.is_absolute():
                model_file = Path(__file__).parent / model_file
            
            if model_file.exists():
                self.model = YOLO(str(model_file))
                self.logger.info(f"YOLO 모델 로드 성공: {model_file}")
            else:
                # 프로젝트 루트의 기본 모델 시도
                project_root = Path(__file__).parent.parent.parent.parent
                default_model = project_root / "yolov8n.pt"
                
                if default_model.exists():
                    self.model = YOLO(str(default_model))
                    self.logger.warning(f"학습된 모델을 찾을 수 없습니다. 기본 모델 사용: {default_model}")
                else:
                    # Ultralytics에서 자동 다운로드
                    self.model = YOLO("yolov8n.pt")
                    self.logger.warning(f"기본 모델을 찾을 수 없습니다. 자동 다운로드: yolov8n.pt")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            try:
                self.model = YOLO("yolov8n.pt")
                self.logger.info("기본 모델로 폴백")
            except Exception as fallback_error:
                self.logger.critical(f"모델 로드 완전 실패: {fallback_error}")
                raise RuntimeError("YOLO 모델을 로드할 수 없습니다.") from fallback_error
    
    def detect_fire(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int, float, str]]]:
        """
        프레임에서 화재 탐지
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (화재 검출 여부, 탐지 결과 리스트 [(x1, y1, x2, y2, confidence, class_name), ...])
        """
        try:
            # 입력 검증
            if frame is None:
                self.logger.warning("입력 프레임이 None입니다.")
                return False, []
            
            if not isinstance(frame, np.ndarray):
                self.logger.warning("입력이 numpy 배열이 아닙니다.")
                return False, []
            
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.warning("입력 프레임이 3채널 이미지가 아닙니다.")
                return False, []
            
            # YOLO 모델로 탐지 실행
            results = self.model(frame, conf=self.confidence_threshold, device=self.device, verbose=False)
            
            detections = []
            fire_detected = False
            
            # 화재 관련 클래스 키워드 (소문자로 통일)
            fire_keywords = ['fire', 'smoke', 'flame', 'burning']
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        try:
                            # 바운딩 박스 좌표
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # 좌표 유효성 검증
                            h, w = frame.shape[:2]
                            x1 = max(0, min(x1, w))
                            y1 = max(0, min(y1, h))
                            x2 = max(0, min(x2, w))
                            y2 = max(0, min(y2, h))
                            
                            # 클래스 정보
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = result.names.get(class_id, "unknown")
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            # 화재 관련 클래스만 필터링
                            if any(keyword in class_name.lower() for keyword in fire_keywords):
                                detections.append((x1, y1, x2, y2, confidence, class_name))
                                fire_detected = True
                        except Exception as box_error:
                            self.logger.warning(f"바운딩 박스 처리 중 오류: {box_error}")
                            continue
            
            if fire_detected:
                self.logger.info(f"YOLO 화재 탐지: {len(detections)}개 객체")
            
            return fire_detected, detections
            
        except Exception as e:
            self.logger.error(f"YOLO 탐지 중 오류 발생: {e}")
            return False, []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float, str]]) -> np.ndarray:
        """
        탐지된 화재 영역을 프레임에 그리기
        
        Args:
            frame: 원본 프레임
            detections: 탐지 결과 리스트
            
        Returns:
            시각화된 프레임
        """
        if frame is None or len(detections) == 0:
            return frame.copy() if frame is not None else None
        
        result_frame = frame.copy()
        
        for x1, y1, x2, y2, confidence, class_name in detections:
            try:
                # 좌표 유효성 검증
                h, w = result_frame.shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # 바운딩 박스 그리기
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 라벨 그리기 (텍스트가 프레임 밖으로 나가지 않도록 조정)
                label = f"{class_name} {confidence:.2f}"
                label_y = max(y1 - 10, 20)  # 최소 20픽셀 여백
                cv2.putText(result_frame, label, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception as draw_error:
                self.logger.warning(f"그리기 중 오류: {draw_error}")
                continue
        
        return result_frame
    
    def update_confidence_threshold(self, threshold: float):
        """
        신뢰도 임계값 업데이트
        
        Args:
            threshold: 새로운 신뢰도 임계값 (0.0 ~ 1.0)
            
        Raises:
            ValueError: 임계값이 유효 범위를 벗어난 경우
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"신뢰도 임계값은 0.0과 1.0 사이여야 합니다: {threshold}")
        
        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        self.logger.info(f"신뢰도 임계값 업데이트: {old_threshold:.2f} -> {threshold:.2f}")
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        try:
            info = {
                "model_name": getattr(self.model, 'model_name', 'YOLOv8'),
                "confidence_threshold": self.confidence_threshold,
                "device": self.device,
                "classes": []
            }
            
            # 클래스 정보 추출
            if hasattr(self.model, 'names') and self.model.names:
                info["classes"] = list(self.model.names.values())
                info["num_classes"] = len(self.model.names)
            
            # 모델 경로 정보 추가
            if hasattr(self.model, 'ckpt_path') and self.model.ckpt_path:
                info["model_path"] = str(self.model.ckpt_path)
            
            return info
        except Exception as e:
            self.logger.error(f"모델 정보 조회 실패: {e}")
            return {
                "error": str(e),
                "confidence_threshold": self.confidence_threshold,
                "device": self.device
            }
