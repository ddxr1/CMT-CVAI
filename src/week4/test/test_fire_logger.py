"""
화재 로거 모듈 테스트
"""

import sys
import os
import json
import csv
import tempfile
import shutil
from datetime import datetime
import pytest

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 상대 경로로 import
from fire_logger import FireLogger

class TestFireLogger:
    """화재 로거 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.logger = FireLogger(
            log_dir=self.temp_dir,
            log_format="both",
            max_log_files=5
        )
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_initialization(self):
        """로거 초기화 테스트"""
        assert self.logger is not None
        assert self.logger.log_dir == self.temp_dir
        assert self.logger.log_format == "both"
        assert self.logger.max_log_files == 5
        assert os.path.exists(self.temp_dir)
    
    def test_log_fire_detection_fire_detected(self):
        """화재 감지 로그 테스트"""
        self.logger.log_fire_detection(
            detection_method="OpenCV",
            fire_detected=True,
            confidence=0.8,
            location={"x": 100, "y": 200, "width": 50, "height": 60},
            frame_size=(640, 480),
            additional_info={"test": "data"}
        )
        
        # JSON 로그 파일 확인
        json_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        assert len(json_files) == 1
        
        with open(os.path.join(self.temp_dir, json_files[0]), 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        assert len(logs) == 1
        assert logs[0]["detection_method"] == "OpenCV"
        assert logs[0]["fire_detected"] == True
        assert logs[0]["confidence"] == 0.8
        assert logs[0]["location"] == {"x": 100, "y": 200, "width": 50, "height": 60}
        assert logs[0]["frame_size"] == [640, 480]
        assert logs[0]["additional_info"] == {"test": "data"}
    
    def test_log_fire_detection_no_fire(self):
        """화재 미감지 로그 테스트"""
        self.logger.log_fire_detection(
            detection_method="YOLO",
            fire_detected=False,
            confidence=0.2,
            location={},
            frame_size=(320, 240),
            additional_info={}
        )
        
        # JSON 로그 파일 확인
        json_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        assert len(json_files) == 1
        
        with open(os.path.join(self.temp_dir, json_files[0]), 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        assert len(logs) == 1
        assert logs[0]["detection_method"] == "YOLO"
        assert logs[0]["fire_detected"] == False
        assert logs[0]["confidence"] == 0.2
    
    def test_csv_log_creation(self):
        """CSV 로그 생성 테스트"""
        self.logger.log_fire_detection(
            detection_method="Hybrid",
            fire_detected=True,
            confidence=0.9
        )
        
        # CSV 로그 파일 확인
        csv_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        assert len(csv_files) == 1
        
        with open(os.path.join(self.temp_dir, csv_files[0]), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # 헤더 + 1개 데이터 행
        assert len(rows) == 2
        assert rows[0] == ['timestamp', 'detection_method', 'fire_detected', 
                          'confidence', 'location', 'frame_size', 'additional_info']
        assert rows[1][1] == "Hybrid"  # detection_method
        assert rows[1][2] == "True"    # fire_detected
    
    def test_get_fire_statistics(self):
        """화재 통계 조회 테스트"""
        # 여러 로그 추가
        for i in range(5):
            self.logger.log_fire_detection(
                detection_method="OpenCV" if i % 2 == 0 else "YOLO",
                fire_detected=i % 3 == 0,  # 3개 중 1개만 화재 감지
                confidence=0.5 + i * 0.1
            )
        
        stats = self.logger.get_fire_statistics(hours=24)
        
        assert "total_detections" in stats
        assert "fire_detections" in stats
        assert "fire_rate" in stats
        assert "method_statistics" in stats
        
        assert stats["total_detections"] == 5
        assert stats["fire_detections"] == 2  # 0, 3번째만 화재 감지
        assert stats["fire_rate"] == 0.4  # 2/5
    
    def test_get_fire_statistics_no_logs(self):
        """로그가 없을 때 통계 조회 테스트"""
        stats = self.logger.get_fire_statistics(hours=24)
        
        assert "error" in stats
        assert "로그 파일이 없습니다" in stats["error"]
    
    def test_cleanup_old_logs(self):
        """오래된 로그 정리 테스트"""
        # max_log_files보다 많은 로그 파일 생성 시뮬레이션
        for i in range(10):
            # 임시 로그 파일 생성
            temp_file = os.path.join(self.temp_dir, f"fire_detection_{i:03d}.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # 정리 실행
        self.logger.cleanup_old_logs()
        
        # max_log_files(5)개만 남아야 함
        json_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        assert len(json_files) <= 5
    
    def test_log_with_minimal_data(self):
        """최소 데이터로 로그 테스트"""
        self.logger.log_fire_detection(
            detection_method="Test",
            fire_detected=True
        )
        
        # 기본값들이 올바르게 설정되는지 확인
        json_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        with open(os.path.join(self.temp_dir, json_files[0]), 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        log = logs[0]
        assert log["confidence"] == 0.0
        assert log["location"] == {}
        assert log["frame_size"] == [0, 0]
        assert log["additional_info"] == {}
    
    def test_different_log_formats(self):
        """다양한 로그 형식 테스트"""
        # 새로운 임시 디렉토리 생성
        json_temp_dir = tempfile.mkdtemp()
        csv_temp_dir = tempfile.mkdtemp()
        
        try:
            # JSON만
            json_logger = FireLogger(log_dir=json_temp_dir, log_format="json")
            json_logger.log_fire_detection("Test", True, 0.5)
            
            json_files = [f for f in os.listdir(json_temp_dir) if f.endswith('.json')]
            csv_files = [f for f in os.listdir(json_temp_dir) if f.endswith('.csv')]
            
            assert len(json_files) == 1
            assert len(csv_files) == 0
            
            # CSV만
            csv_logger = FireLogger(log_dir=csv_temp_dir, log_format="csv")
            csv_logger.log_fire_detection("Test", True, 0.5)
            
            json_files = [f for f in os.listdir(csv_temp_dir) if f.endswith('.json')]
            csv_files = [f for f in os.listdir(csv_temp_dir) if f.endswith('.csv')]
            
            assert len(json_files) == 0
            assert len(csv_files) == 1
            
        finally:
            # 임시 디렉토리 정리
            shutil.rmtree(json_temp_dir, ignore_errors=True)
            shutil.rmtree(csv_temp_dir, ignore_errors=True)

def run_tests():
    """테스트 실행 함수"""
    print("🔥 화재 로거 테스트 실행")
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
