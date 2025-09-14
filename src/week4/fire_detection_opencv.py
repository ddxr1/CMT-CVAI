"""
OpenCV 기반 화염 검출 모듈
HSV 색공간을 활용하여 화염의 색상 특징을 기반으로 화재를 탐지합니다.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

class FireDetectorOpenCV:
    """OpenCV를 사용한 화염 검출 클래스"""
    
    def __init__(self, 
                 lower_hsv: Tuple[int, int, int] = (0, 50, 50),
                 upper_hsv: Tuple[int, int, int] = (35, 255, 255),
                 min_area: int = 1000,
                 confidence_threshold: float = 0.3):
        """
        화염 검출기 초기화
        
        Args:
            lower_hsv: HSV 색공간에서 화염의 하한값 (H, S, V)
            upper_hsv: HSV 색공간에서 화염의 상한값 (H, S, V)
            min_area: 최소 화염 영역 크기 (픽셀)
            confidence_threshold: 화염 검출 신뢰도 임계값
        """
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.min_area = min_area
        self.confidence_threshold = confidence_threshold
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 전처리
        
        Args:
            frame: 입력 프레임
            
        Returns:
            전처리된 프레임
        """
        # 가우시안 블러 적용 (노이즈 제거)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def detect_fire_regions(self, hsv_frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        HSV 프레임에서 화염 영역 검출
        
        Args:
            hsv_frame: HSV 색공간 프레임
            
        Returns:
            화염 영역 리스트 [(x, y, w, h, confidence), ...]
        """
        # HSV 범위에 따른 마스크 생성
        mask = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 최소 영역 크기 확인
            if area > self.min_area:
                # 바운딩 박스 계산
                x, y, w, h = cv2.boundingRect(contour)
                
                # 신뢰도 계산 (면적 기반)
                confidence = min(area / (w * h), 1.0)
                
                if confidence >= self.confidence_threshold:
                    fire_regions.append((x, y, w, h, confidence))
        
        return fire_regions
    
    def detect_fire(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int, float]]]:
        """
        프레임에서 화재 검출
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (화재 검출 여부, 화염 영역 리스트)
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
            
            # 프레임 전처리
            hsv_frame = self.preprocess_frame(frame)
            
            # 화염 영역 검출
            fire_regions = self.detect_fire_regions(hsv_frame)
            
            # 화재 검출 여부 판단
            fire_detected = len(fire_regions) > 0
            
            if fire_detected:
                self.logger.info(f"화재 검출됨: {len(fire_regions)}개 영역")
            
            return fire_detected, fire_regions
            
        except Exception as e:
            self.logger.error(f"화재 검출 중 오류 발생: {e}")
            return False, []
    
    def draw_detections(self, frame: np.ndarray, fire_regions: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        검출된 화염 영역을 프레임에 그리기
        
        Args:
            frame: 원본 프레임
            fire_regions: 화염 영역 리스트
            
        Returns:
            시각화된 프레임
        """
        result_frame = frame.copy()
        
        for x, y, w, h, confidence in fire_regions:
            # 바운딩 박스 그리기
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 라벨 그리기
            label = f"Fire {confidence:.2f}"
            cv2.putText(result_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result_frame
    
    def update_parameters(self, 
                         lower_hsv: Optional[Tuple[int, int, int]] = None,
                         upper_hsv: Optional[Tuple[int, int, int]] = None,
                         min_area: Optional[int] = None,
                         confidence_threshold: Optional[float] = None):
        """
        검출 파라미터 업데이트
        
        Args:
            lower_hsv: HSV 하한값
            upper_hsv: HSV 상한값
            min_area: 최소 영역 크기
            confidence_threshold: 신뢰도 임계값
        """
        if lower_hsv is not None:
            self.lower_hsv = np.array(lower_hsv)
        if upper_hsv is not None:
            self.upper_hsv = np.array(upper_hsv)
        if min_area is not None:
            self.min_area = min_area
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        self.logger.info("검출 파라미터가 업데이트되었습니다.")
