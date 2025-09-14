"""
실시간 화재 감지기 모듈 테스트
"""

import sys
import os
import numpy as np
import cv2
import pytest
import tempfile
import shutil

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 상대 경로로 import
from real_time_fire_detector import RealTimeFireDetector

class TestRealTimeFireDetector:
    """실시간 화재 감지기 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.detector = RealTimeFireDetector(
            use_opencv=True,
            use_yolo=True,
            hybrid_mode=True,
            confidence_threshold=0.5,
            log_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detector_initialization(self):
        """감지기 초기화 테스트"""
        assert self.detector is not None
        assert self.detector.use_opencv == True
        assert self.detector.use_yolo == True
        assert self.detector.hybrid_mode == True
        assert self.detector.confidence_threshold == 0.5
        assert self.detector.is_running == False
        assert self.detector.total_frames == 0
        assert self.detector.fire_detections == 0
    
    def test_detect_fire_opencv_only(self):
        """OpenCV만 사용한 화재 감지 테스트"""
        detector = RealTimeFireDetector(
            use_opencv=True,
            use_yolo=False,
            hybrid_mode=False,
            confidence_threshold=0.3
        )
        
        # 빨간색 이미지 생성
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:70, 30:70] = [0, 0, 255]  # BGR 형식으로 빨간색
        
        fire_detected, detection_info = detector._detect_fire(image)
        
        assert isinstance(fire_detected, bool)
        assert isinstance(detection_info, dict)
        assert "opencv" in detection_info
        assert "yolo" in detection_info
        assert "hybrid" in detection_info
    
    def test_detect_fire_yolo_only(self):
        """YOLO만 사용한 화재 감지 테스트"""
        detector = RealTimeFireDetector(
            use_opencv=False,
            use_yolo=True,
            hybrid_mode=False,
            confidence_threshold=0.5
        )
        
        # 랜덤 이미지 생성
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        fire_detected, detection_info = detector._detect_fire(image)
        
        assert isinstance(fire_detected, bool)
        assert isinstance(detection_info, dict)
        assert "opencv" in detection_info
        assert "yolo" in detection_info
        assert "hybrid" in detection_info
    
    def test_detect_fire_hybrid_mode(self):
        """하이브리드 모드 화재 감지 테스트"""
        # 빨간색 이미지 생성
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:70, 30:70] = [0, 0, 255]  # BGR 형식으로 빨간색
        
        fire_detected, detection_info = self.detector._detect_fire(image)
        
        assert isinstance(fire_detected, bool)
        assert isinstance(detection_info, dict)
        assert "opencv" in detection_info
        assert "yolo" in detection_info
        assert "hybrid" in detection_info
        
        # 하이브리드 모드에서는 두 방법의 결과를 결합
        assert "detected" in detection_info["hybrid"]
        assert "confidence" in detection_info["hybrid"]
    
    def test_visualize_results(self):
        """결과 시각화 테스트"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detection_info = {
            "opencv": {"detected": True, "regions": [(10, 10, 20, 20, 0.8)], "confidence": 0.8},
            "yolo": {"detected": False, "detections": [], "confidence": 0.0},
            "hybrid": {"detected": True, "confidence": 0.8}
        }
        
        result = self.detector._visualize_results(image, True, detection_info)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert not np.array_equal(result, image)  # 시각화된 결과가 원본과 달라야 함
    
    def test_log_detection(self):
        """감지 로그 기록 테스트"""
        detection_info = {
            "opencv": {"detected": True, "regions": [(10, 10, 20, 20, 0.8)], "confidence": 0.8},
            "yolo": {"detected": False, "detections": [], "confidence": 0.0},
            "hybrid": {"detected": True, "confidence": 0.8}
        }
        
        # 로그 기록
        self.detector._log_detection(True, detection_info)
        
        # 로그 파일 확인
        log_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        assert len(log_files) == 1
    
    def test_get_statistics(self):
        """통계 조회 테스트"""
        # 가상의 통계 설정
        self.detector.total_frames = 100
        self.detector.fire_detections = 5
        
        stats = self.detector.get_statistics()
        
        assert "total_frames" in stats
        assert "fire_detections" in stats
        assert "detection_rate" in stats
        assert "log_statistics" in stats
        
        assert stats["total_frames"] == 100
        assert stats["fire_detections"] == 5
        assert stats["detection_rate"] == 0.05  # 5/100
    
    def test_get_statistics_no_frames(self):
        """프레임이 없을 때 통계 조회 테스트"""
        stats = self.detector.get_statistics()
        
        assert stats["total_frames"] == 0
        assert stats["fire_detections"] == 0
        assert stats["detection_rate"] == 0
    
    def test_save_current_frame(self):
        """현재 프레임 저장 테스트"""
        # 가상의 현재 프레임 설정
        self.detector.current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 프레임 저장 (실제로는 파일이 생성되지만 테스트에서는 예외 없이 실행되는지 확인)
        try:
            self.detector._save_current_frame()
            # 예외가 발생하지 않으면 성공
            assert True
        except Exception as e:
            pytest.fail(f"프레임 저장 중 예외 발생: {e}")
    
    def test_print_detection_info(self):
        """감지 정보 출력 테스트"""
        # 로그 기록
        detection_info = {
            "opencv": {"detected": True, "regions": [], "confidence": 0.8},
            "yolo": {"detected": False, "detections": [], "confidence": 0.0},
            "hybrid": {"detected": True, "confidence": 0.8}
        }
        self.detector._log_detection(True, detection_info)
        
        # 감지 정보 출력 (예외 없이 실행되는지 확인)
        try:
            self.detector._print_detection_info()
            assert True
        except Exception as e:
            pytest.fail(f"감지 정보 출력 중 예외 발생: {e}")
    
    def test_detection_with_different_image_sizes(self):
        """다양한 이미지 크기로 감지 테스트"""
        sizes = [(100, 100), (320, 240), (640, 480)]
        
        for height, width in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            fire_detected, detection_info = self.detector._detect_fire(image)
            
            assert isinstance(fire_detected, bool)
            assert isinstance(detection_info, dict)
    
    def test_detection_with_invalid_input(self):
        """잘못된 입력에 대한 테스트"""
        # None 입력
        fire_detected, detection_info = self.detector._detect_fire(None)
        assert not fire_detected
        assert isinstance(detection_info, dict)
        
        # 잘못된 차원의 배열
        invalid_image = np.zeros((100, 100), dtype=np.uint8)  # 2D 배열
        fire_detected, detection_info = self.detector._detect_fire(invalid_image)
        assert not fire_detected
        assert isinstance(detection_info, dict)
    
    def test_detector_with_different_configurations(self):
        """다양한 설정으로 감지기 테스트"""
        configs = [
            {"use_opencv": True, "use_yolo": False, "hybrid_mode": False},
            {"use_opencv": False, "use_yolo": True, "hybrid_mode": False},
            {"use_opencv": True, "use_yolo": True, "hybrid_mode": True},
        ]
        
        for config in configs:
            detector = RealTimeFireDetector(**config)
            assert detector.use_opencv == config["use_opencv"]
            assert detector.use_yolo == config["use_yolo"]
            assert detector.hybrid_mode == config["hybrid_mode"]

def run_tests():
    """테스트 실행 함수"""
    print("🔥 실시간 화재 감지기 테스트 실행")
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
