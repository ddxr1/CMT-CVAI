"""
화재 감지 로그 기록 모듈
화재 발생 시 발생 시간, 위치 정보, 신뢰도 등을 로그로 기록합니다.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import csv

class FireLogger:
    """화재 감지 로그 관리 클래스"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_format: str = "both",  # "json", "csv", "both"
                 max_log_files: int = 10):
        """
        화재 로거 초기화
        
        Args:
            log_dir: 로그 파일 저장 디렉토리
            log_format: 로그 형식 ("json", "csv", "both")
            max_log_files: 최대 로그 파일 개수
        """
        self.log_dir = log_dir
        self.log_format = log_format
        self.max_log_files = max_log_files
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 로그 파일 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_file = os.path.join(log_dir, f"fire_detection_{timestamp}.json")
        self.csv_log_file = os.path.join(log_dir, f"fire_detection_{timestamp}.csv")
        
        # CSV 헤더 작성
        if log_format in ["csv", "both"]:
            self._write_csv_header()
    
    def setup_logging(self):
        """로깅 설정"""
        # 콘솔 로거 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # 파일 로거 설정
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, "fire_detection.log"),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 루트 로거 설정
        self.logger = logging.getLogger("FireDetection")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _write_csv_header(self):
        """CSV 파일 헤더 작성"""
        try:
            with open(self.csv_log_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'detection_method', 'fire_detected', 
                    'confidence', 'location', 'frame_size', 'additional_info'
                ])
        except Exception as e:
            self.logger.error(f"CSV 헤더 작성 실패: {e}")
    
    def log_fire_detection(self, 
                          detection_method: str,
                          fire_detected: bool,
                          confidence: float = 0.0,
                          location: Optional[Dict[str, int]] = None,
                          frame_size: Optional[Tuple[int, int]] = None,
                          additional_info: Optional[Dict[str, Any]] = None):
        """
        화재 감지 로그 기록
        
        Args:
            detection_method: 감지 방법 ("OpenCV", "YOLO", "Hybrid")
            fire_detected: 화재 검출 여부
            confidence: 신뢰도
            location: 위치 정보 {"x": x, "y": y, "width": w, "height": h}
            frame_size: 프레임 크기 (width, height)
            additional_info: 추가 정보
        """
        timestamp = datetime.now()
        
        # 로그 데이터 구성
        log_data = {
            "timestamp": timestamp.isoformat(),
            "detection_method": detection_method,
            "fire_detected": fire_detected,
            "confidence": confidence,
            "location": location or {},
            "frame_size": frame_size or (0, 0),
            "additional_info": additional_info or {}
        }
        
        # 콘솔 로그 출력
        if fire_detected:
            self.logger.warning(
                f"화재 감지! 방법: {detection_method}, "
                f"신뢰도: {confidence:.2f}, 위치: {location}"
            )
        else:
            self.logger.info(
                f"정상 상태 - 방법: {detection_method}, 신뢰도: {confidence:.2f}"
            )
        
        # 파일 로그 기록
        if self.log_format in ["json", "both"]:
            self._write_json_log(log_data)
        
        if self.log_format in ["csv", "both"]:
            self._write_csv_log(log_data)
    
    def _write_json_log(self, log_data: Dict[str, Any]):
        """JSON 로그 파일에 기록"""
        try:
            # 기존 로그 파일이 있으면 읽어서 추가
            logs = []
            if os.path.exists(self.json_log_file):
                with open(self.json_log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            
            # 새 로그 추가
            logs.append(log_data)
            
            # 파일에 저장
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"JSON 로그 기록 실패: {e}")
    
    def _write_csv_log(self, log_data: Dict[str, Any]):
        """CSV 로그 파일에 기록"""
        try:
            with open(self.csv_log_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    log_data["timestamp"],
                    log_data["detection_method"],
                    log_data["fire_detected"],
                    log_data["confidence"],
                    str(log_data["location"]),
                    str(log_data["frame_size"]),
                    str(log_data["additional_info"])
                ])
        except Exception as e:
            self.logger.error(f"CSV 로그 기록 실패: {e}")
    
    def get_fire_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        화재 감지 통계 조회
        
        Args:
            hours: 조회할 시간 범위 (시간)
            
        Returns:
            통계 정보 딕셔너리
        """
        try:
            if not os.path.exists(self.json_log_file):
                return {"error": "로그 파일이 없습니다."}
            
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # 시간 필터링
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_logs = [
                log for log in logs 
                if datetime.fromisoformat(log["timestamp"]).timestamp() > cutoff_time
            ]
            
            # 통계 계산
            total_detections = len(recent_logs)
            fire_detections = len([log for log in recent_logs if log["fire_detected"]])
            
            # 방법별 통계
            method_stats = {}
            for log in recent_logs:
                method = log["detection_method"]
                if method not in method_stats:
                    method_stats[method] = {"total": 0, "fire": 0}
                method_stats[method]["total"] += 1
                if log["fire_detected"]:
                    method_stats[method]["fire"] += 1
            
            return {
                "period_hours": hours,
                "total_detections": total_detections,
                "fire_detections": fire_detections,
                "fire_rate": fire_detections / total_detections if total_detections > 0 else 0,
                "method_statistics": method_stats
            }
            
        except Exception as e:
            self.logger.error(f"통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def cleanup_old_logs(self):
        """오래된 로그 파일 정리"""
        try:
            log_files = [f for f in os.listdir(self.log_dir) if f.startswith("fire_detection_")]
            log_files.sort()
            
            # 최대 파일 개수 초과 시 오래된 파일 삭제
            if len(log_files) > self.max_log_files:
                files_to_delete = log_files[:-self.max_log_files]
                for file in files_to_delete:
                    os.remove(os.path.join(self.log_dir, file))
                    self.logger.info(f"오래된 로그 파일 삭제: {file}")
                    
        except Exception as e:
            self.logger.error(f"로그 정리 실패: {e}")
