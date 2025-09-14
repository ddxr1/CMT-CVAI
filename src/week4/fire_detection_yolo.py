"""
YOLOv8 기반 화재 탐지 모듈
사전 학습된 YOLOv8 모델을 사용하여 화재를 탐지합니다.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
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
            # 절대 경로로 변환
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.logger.info(f"YOLO 모델 로드 성공: {model_path}")
            else:
                # 프로젝트 루트의 기본 모델 사용
                default_model = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "yolov8n.pt")
                if os.path.exists(default_model):
                    self.model = YOLO(default_model)
                    self.logger.warning(f"학습된 모델을 찾을 수 없습니다. 기본 모델 사용: {default_model}")
                else:
                    # Ultralytics에서 자동 다운로드
                    self.model = YOLO("yolov8n.pt")
                    self.logger.warning(f"기본 모델을 찾을 수 없습니다. 자동 다운로드: yolov8n.pt")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            self.model = YOLO("yolov8n.pt")
    
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
            results = self.model(frame, conf=self.confidence_threshold, device=self.device)
            
            detections = []
            fire_detected = False
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 클래스 정보
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # 화재 관련 클래스만 필터링 (Fire, Smoke 등)
                        # 학습된 모델의 경우 클래스명을 확인하고, 기본 모델의 경우 모든 객체를 화재로 간주
                        if ('fire' in class_name.lower() or 'smoke' in class_name.lower() or 
                            class_name.lower() in ['fire', 'smoke', 'flame'] or
                            len(self.model.names) == 80):  # COCO 데이터셋의 경우 모든 객체 허용
                            detections.append((x1, y1, x2, y2, confidence, class_name))
                            fire_detected = True
            
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
        result_frame = frame.copy()
        
        for x1, y1, x2, y2, confidence, class_name in detections:
            # 바운딩 박스 그리기
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 라벨 그리기
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result_frame
    
    def update_confidence_threshold(self, threshold: float):
        """
        신뢰도 임계값 업데이트
        
        Args:
            threshold: 새로운 신뢰도 임계값
        """
        self.confidence_threshold = threshold
        self.logger.info(f"신뢰도 임계값 업데이트: {threshold}")
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        try:
            return {
                "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "YOLOv8",
                "confidence_threshold": self.confidence_threshold,
                "device": self.device,
                "classes": list(self.model.names.values()) if hasattr(self.model, 'names') else []
            }
        except Exception as e:
            self.logger.error(f"모델 정보 조회 실패: {e}")
            return {}
