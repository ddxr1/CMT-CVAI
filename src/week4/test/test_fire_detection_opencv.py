"""
OpenCV 화재 감지 모듈 테스트
"""

import sys
import os
import numpy as np
import cv2
import pytest

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 상대 경로로 import
from fire_detection_opencv import FireDetectorOpenCV

class TestFireDetectorOpenCV:
    """OpenCV 화재 감지기 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        self.detector = FireDetectorOpenCV(
            lower_hsv=(0, 50, 50),
            upper_hsv=(35, 255, 255),
            min_area=100,
            confidence_threshold=0.3
        )
    
    def test_detector_initialization(self):
        """감지기 초기화 테스트"""
        assert self.detector is not None
        assert self.detector.confidence_threshold == 0.3
        assert self.detector.min_area == 100
        assert np.array_equal(self.detector.lower_hsv, np.array([0, 50, 50]))
        assert np.array_equal(self.detector.upper_hsv, np.array([35, 255, 255]))
    
    def test_detect_fire_no_fire(self):
        """화재가 없는 이미지 테스트"""
        # 검은색 이미지 생성
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        fire_detected, regions = self.detector.detect_fire(image)
        
        assert not fire_detected
        assert len(regions) == 0
    
    def test_detect_fire_with_fire_color(self):
        """화재 색상이 있는 이미지 테스트"""
        # 빨간색 이미지 생성 (화재 색상)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:70, 30:70] = [0, 0, 255]  # BGR 형식으로 빨간색
        
        fire_detected, regions = self.detector.detect_fire(image)
        
        # 화재 색상이 감지되어야 함
        assert fire_detected
        assert len(regions) > 0
    
    def test_preprocess_frame(self):
        """프레임 전처리 테스트"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed = self.detector.preprocess_frame(image)
        
        assert processed.shape == image.shape
        assert processed.dtype == np.uint8
    
    def test_detect_fire_regions(self):
        """화재 영역 감지 테스트"""
        # HSV 이미지 생성
        hsv_image = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_image[30:70, 30:70] = [10, 255, 255]  # 주황색 (화재 색상)
        
        regions = self.detector.detect_fire_regions(hsv_image)
        
        assert isinstance(regions, list)
        if len(regions) > 0:
            for region in regions:
                assert len(region) == 5  # (x, y, w, h, confidence)
                assert all(isinstance(x, (int, float)) for x in region)
    
    def test_draw_detections(self):
        """감지 결과 그리기 테스트"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regions = [(10, 10, 20, 20, 0.8), (50, 50, 30, 30, 0.9)]
        
        result = self.detector.draw_detections(image, regions)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert not np.array_equal(result, image)  # 그려진 결과가 원본과 달라야 함
    
    def test_update_parameters(self):
        """파라미터 업데이트 테스트"""
        original_lower = self.detector.lower_hsv.copy()
        original_upper = self.detector.upper_hsv.copy()
        
        # 파라미터 업데이트
        self.detector.update_parameters(
            lower_hsv=(5, 60, 60),
            upper_hsv=(30, 250, 250),
            min_area=200,
            confidence_threshold=0.5
        )
        
        assert not np.array_equal(self.detector.lower_hsv, original_lower)
        assert not np.array_equal(self.detector.upper_hsv, original_upper)
        assert self.detector.min_area == 200
        assert self.detector.confidence_threshold == 0.5
    
    def test_detect_fire_with_small_area(self):
        """작은 영역은 감지되지 않아야 함"""
        # 매우 작은 빨간색 영역 생성 (5x5 = 25 픽셀)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[47:52, 47:52] = [0, 0, 255]  # 5x5 영역
        
        fire_detected, regions = self.detector.detect_fire(image)
        
        # min_area가 100이므로 5x5=25 픽셀 영역은 감지되지 않아야 함
        assert not fire_detected
        assert len(regions) == 0
    
    def test_detect_fire_with_large_area(self):
        """큰 영역은 감지되어야 함"""
        # 큰 빨간색 영역 생성
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150] = [0, 0, 255]  # 100x100 영역
        
        fire_detected, regions = self.detector.detect_fire(image)
        
        # min_area가 100이므로 100x100 영역은 감지되어야 함
        assert fire_detected
        assert len(regions) > 0

def run_tests():
    """테스트 실행 함수"""
    print("🔥 OpenCV 화재 감지기 테스트 실행")
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
