"""
YOLO 화재 감지 모듈 테스트
"""

import sys
import os
import numpy as np
import cv2
import pytest

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 상대 경로로 import
from fire_detection_yolo import FireDetectorYOLO

class TestFireDetectorYOLO:
    """YOLO 화재 감지기 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        self.detector = FireDetectorYOLO(
            model_path="../week3/runs/detect/train/weights/best.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
    
    def test_detector_initialization(self):
        """감지기 초기화 테스트"""
        assert self.detector is not None
        assert self.detector.confidence_threshold == 0.5
        assert self.detector.device == "cpu"
        assert self.detector.model is not None
    
    def test_detect_fire_no_objects(self):
        """객체가 없는 이미지 테스트"""
        # 검은색 이미지 생성
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        fire_detected, detections = self.detector.detect_fire(image)
        
        assert not fire_detected
        assert len(detections) == 0
    
    def test_detect_fire_with_objects(self):
        """객체가 있는 이미지 테스트"""
        # 랜덤 이미지 생성 (객체가 있을 수 있음)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        fire_detected, detections = self.detector.detect_fire(image)
        
        assert isinstance(fire_detected, bool)
        assert isinstance(detections, list)
        
        if detections:
            for detection in detections:
                assert len(detection) == 6  # (x1, y1, x2, y2, confidence, class_name)
                assert all(isinstance(x, (int, float, str)) for x in detection)
    
    def test_draw_detections(self):
        """감지 결과 그리기 테스트"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [(10, 10, 30, 30, 0.8, "fire"), (50, 50, 70, 70, 0.9, "smoke")]
        
        result = self.detector.draw_detections(image, detections)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert not np.array_equal(result, image)  # 그려진 결과가 원본과 달라야 함
    
    def test_update_confidence_threshold(self):
        """신뢰도 임계값 업데이트 테스트"""
        original_threshold = self.detector.confidence_threshold
        
        self.detector.update_confidence_threshold(0.7)
        
        assert self.detector.confidence_threshold == 0.7
        assert self.detector.confidence_threshold != original_threshold
    
    def test_get_model_info(self):
        """모델 정보 조회 테스트"""
        info = self.detector.get_model_info()
        
        assert isinstance(info, dict)
        assert "confidence_threshold" in info
        assert "device" in info
        assert "classes" in info
        assert info["confidence_threshold"] == self.detector.confidence_threshold
        assert info["device"] == self.detector.device
    
    def test_detect_fire_with_different_sizes(self):
        """다양한 크기의 이미지 테스트"""
        sizes = [(100, 100), (320, 240), (640, 480), (1280, 720)]
        
        for height, width in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            fire_detected, detections = self.detector.detect_fire(image)
            
            assert isinstance(fire_detected, bool)
            assert isinstance(detections, list)
    
    def test_detect_fire_with_high_confidence(self):
        """높은 신뢰도로 감지 테스트"""
        # 높은 신뢰도 임계값 설정
        self.detector.update_confidence_threshold(0.9)
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        fire_detected, detections = self.detector.detect_fire(image)
        
        # 높은 신뢰도에서는 감지가 적어야 함
        assert isinstance(fire_detected, bool)
        assert isinstance(detections, list)
    
    def test_detect_fire_with_low_confidence(self):
        """낮은 신뢰도로 감지 테스트"""
        # 낮은 신뢰도 임계값 설정
        self.detector.update_confidence_threshold(0.1)
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        fire_detected, detections = self.detector.detect_fire(image)
        
        # 낮은 신뢰도에서는 감지가 많을 수 있음
        assert isinstance(fire_detected, bool)
        assert isinstance(detections, list)
    
    def test_detect_fire_with_invalid_input(self):
        """잘못된 입력에 대한 테스트"""
        # None 입력
        fire_detected, detections = self.detector.detect_fire(None)
        assert not fire_detected
        assert len(detections) == 0
        
        # 잘못된 차원의 배열
        invalid_image = np.zeros((100, 100), dtype=np.uint8)  # 2D 배열
        fire_detected, detections = self.detector.detect_fire(invalid_image)
        assert not fire_detected
        assert len(detections) == 0

def run_tests():
    """테스트 실행 함수"""
    print("🔥 YOLO 화재 감지기 테스트 실행")
    print("=" * 50)
    
    try:
        import pytest
        result = pytest.main([__file__, "-v", "--tb=short"])
        if result == 0:
            print("\n✅ 모든 테스트가 통과했습니다!")
        else:
            print("\n❌ 일부 테스트가 실패했습니다.")
        return result == 0
    except ImportError:
        print("❌ pytest가 설치되지 않았습니다. 'pip install pytest'를 실행하세요.")
        return False

if __name__ == "__main__":
    run_tests()
