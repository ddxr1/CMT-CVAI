"""
화재 감지 시스템 설정 파일
"""

import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# 모델 경로 설정
YOLO_MODEL_PATH = PROJECT_ROOT / "src" / "week3" / "runs" / "detect" / "train" / "weights" / "best.pt"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "yolov8n.pt"

# 입력/출력 경로 설정
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = BASE_DIR / "logs"

# 비디오 설정
DEFAULT_VIDEO_SOURCE = 0  # 웹캠
DEFAULT_FPS = 30
DEFAULT_FRAME_SIZE = (640, 480)

# OpenCV 화재 감지 설정
OPENCV_CONFIG = {
    "lower_hsv": (0, 50, 50),      # HSV 하한값 (H, S, V)
    "upper_hsv": (35, 255, 255),   # HSV 상한값 (H, S, V)
    "min_area": 1000,              # 최소 화염 영역 크기 (픽셀)
    "confidence_threshold": 0.3,   # 신뢰도 임계값
    "blur_kernel_size": (5, 5),    # 가우시안 블러 커널 크기
    "morphology_kernel_size": (5, 5)  # 모폴로지 연산 커널 크기
}

# YOLO 화재 감지 설정
YOLO_CONFIG = {
    "confidence_threshold": 0.5,   # 신뢰도 임계값
    "device": "cpu",               # 사용할 디바이스 (cpu/cuda)
    "input_size": 640,             # 입력 이미지 크기
    "max_detections": 100          # 최대 탐지 객체 수
}

# 하이브리드 감지 설정
HYBRID_CONFIG = {
    "opencv_weight": 0.4,          # OpenCV 감지 가중치
    "yolo_weight": 0.6,            # YOLO 감지 가중치
    "consensus_threshold": 0.6,    # 합의 임계값
    "high_confidence_multiplier": 1.5  # 높은 신뢰도 배수
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",               # 로그 레벨 (DEBUG/INFO/WARNING/ERROR)
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_rotation": True,         # 로그 파일 로테이션
    "max_file_size": 10 * 1024 * 1024,  # 최대 파일 크기 (10MB)
    "backup_count": 5,             # 백업 파일 개수
    "log_formats": ["json", "csv"]  # 로그 형식
}

# 알림 설정
ALERT_CONFIG = {
    "enable_sound": True,          # 소리 알림 활성화
    "enable_visual": True,         # 시각적 알림 활성화
    "cooldown_seconds": 1.0,       # 알림 쿨다운 시간 (초)
    "alert_duration": 5.0,         # 알림 지속 시간 (초)
    "email_notifications": False,  # 이메일 알림 활성화
    "email_recipients": []         # 이메일 수신자 목록
}

# 성능 설정
PERFORMANCE_CONFIG = {
    "max_fps": 30,                 # 최대 FPS
    "frame_skip": 1,               # 프레임 스킵 (1=모든 프레임 처리)
    "threading": True,             # 멀티스레딩 사용
    "gpu_acceleration": False,     # GPU 가속 사용
    "memory_limit": 1024 * 1024 * 1024  # 메모리 제한 (1GB)
}

# UI 설정
UI_CONFIG = {
    "show_fps": True,              # FPS 표시
    "show_confidence": True,       # 신뢰도 표시
    "show_statistics": True,       # 통계 표시
    "window_title": "Real-time Fire Detection System",
    "status_bar_height": 120,      # 상태바 높이
    "font_scale": 0.6,             # 폰트 크기
    "line_thickness": 2            # 선 두께
}

# 데이터베이스 설정 (선택사항)
DATABASE_CONFIG = {
    "enabled": False,              # 데이터베이스 사용 여부
    "type": "sqlite",              # 데이터베이스 타입
    "host": "localhost",
    "port": 5432,
    "database": "fire_detection",
    "username": "",
    "password": ""
}

def get_config(section: str = None):
    """
    설정 조회
    
    Args:
        section: 설정 섹션명 (None이면 전체 설정 반환)
        
    Returns:
        설정 딕셔너리
    """
    configs = {
        "opencv": OPENCV_CONFIG,
        "yolo": YOLO_CONFIG,
        "hybrid": HYBRID_CONFIG,
        "logging": LOGGING_CONFIG,
        "alert": ALERT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "ui": UI_CONFIG,
        "database": DATABASE_CONFIG
    }
    
    if section:
        return configs.get(section, {})
    return configs

def update_config(section: str, key: str, value):
    """
    설정 업데이트
    
    Args:
        section: 설정 섹션명
        key: 설정 키
        value: 설정 값
    """
    configs = {
        "opencv": OPENCV_CONFIG,
        "yolo": YOLO_CONFIG,
        "hybrid": HYBRID_CONFIG,
        "logging": LOGGING_CONFIG,
        "alert": ALERT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "ui": UI_CONFIG,
        "database": DATABASE_CONFIG
    }
    
    if section in configs and key in configs[section]:
        configs[section][key] = value
        print(f"설정 업데이트: {section}.{key} = {value}")
    else:
        print(f"설정을 찾을 수 없습니다: {section}.{key}")

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [LOG_DIR, OUTPUT_DIR / "fire_detection", OUTPUT_DIR / "videos"]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"디렉토리 생성: {directory}")

def validate_config():
    """설정 유효성 검사"""
    errors = []
    
    # 모델 파일 존재 확인
    if not YOLO_MODEL_PATH.exists() and not DEFAULT_YOLO_MODEL.exists():
        errors.append(f"YOLO 모델 파일을 찾을 수 없습니다: {YOLO_MODEL_PATH}")
    
    # 설정 값 범위 확인
    if OPENCV_CONFIG["confidence_threshold"] < 0 or OPENCV_CONFIG["confidence_threshold"] > 1:
        errors.append("OpenCV 신뢰도 임계값은 0-1 사이여야 합니다")
    
    if YOLO_CONFIG["confidence_threshold"] < 0 or YOLO_CONFIG["confidence_threshold"] > 1:
        errors.append("YOLO 신뢰도 임계값은 0-1 사이여야 합니다")
    
    if errors:
        print("설정 오류:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("설정 유효성 검사 통과")
    return True

if __name__ == "__main__":
    # 설정 테스트
    print("화재 감지 시스템 설정 테스트")
    print("=" * 50)
    
    # 디렉토리 생성
    create_directories()
    
    # 설정 유효성 검사
    validate_config()
    
    # 설정 출력
    print("\n📋 현재 설정:")
    for section, config in get_config().items():
        print(f"\n[{section.upper()}]")
        for key, value in config.items():
            print(f"  {key}: {value}")
