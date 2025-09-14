# 화재 객체 탐지 시스템 v1.0

Computer Vision AI를 활용한 화재 객체 탐지 소프트웨어 프로토타입입니다.

## 프로젝트 개요

- **목표**: 실시간 영상에서 화재를 탐지하고 경고를 출력하는 소프트웨어
- **기술**: OpenCV + YOLOv8 하이브리드 감지 시스템
- **특징**: GUI 인터페이스, 실시간 감지, 로깅 시스템

## 빠른 시작

### 1. 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 실행

```bash
# 메인 프로그램 실행
python main.py
```

### 3. 사용법

1. **GUI 모드**: 그래픽 인터페이스로 쉽게 사용
2. **종료**: 프로그램 종료

## GUI 모드 사용법

1. 프로그램 실행 후 "1" 선택
2. 비디오 소스 선택 (웹캠 또는 파일)
3. 감지 모드 선택 (OpenCV, YOLO, 하이브리드)
4. 신뢰도 임계값 조정
5. "결과 영상 저장" 체크박스 선택 (선택사항)
6. "감지 시작" 버튼 클릭

### GUI 기능

- **실시간 비디오 표시**: 감지 결과가 실시간으로 표시
- **통계 정보**: 프레임 수, 감지 수, 감지율 표시
- **다양한 감지 모드**: OpenCV, YOLO, 하이브리드 선택 가능
- **신뢰도 조정**: 실시간으로 감지 민감도 조정
- **자동 영상 종료 처리**: 영상이 끝나면 자동으로 감지 중지
- **결과 영상 저장**: bounding box가 포함된 감지 결과 영상을 MP4 형식으로 저장

## 감지 모드

### OpenCV 모드
- HSV 색공간 기반 화염 색상 검출
- 빠른 처리 속도
- 색상 기반으로만 판단

### YOLO 모드
- AI 모델 기반 화재 객체 탐지
- 높은 정확도
- 더 많은 컴퓨팅 리소스 필요

### 하이브리드 모드
- OpenCV와 YOLO를 결합
- 두 방법의 장점을 활용
- 가장 안정적인 감지

## 통계 및 로그

- **실시간 통계**: GUI에서 실시간으로 감지 통계 확인
- **로그 파일**: `logs/` 폴더에 상세 로그 저장 (CSV 형식)
- **결과 영상**: `outputs/` 폴더에 bounding box가 포함된 감지 결과 영상 저장
- **UTF-8 인코딩**: 한글 로그 메시지 정상 표시

## Unit Test

각 모듈별로 독립적인 Unit Test를 제공합니다.

### 테스트 실행

```bash
# 개별 테스트 실행
python test/test_fire_detection_opencv.py
python test/test_fire_detection_yolo.py
python test/test_fire_logger.py
python test/test_real_time_fire_detector.py

# pytest로 실행
pytest test/ -v
```

### 테스트 파일

- `test_fire_detection_opencv.py`: OpenCV 감지기 테스트
- `test_fire_detection_yolo.py`: YOLO 감지기 테스트
- `test_fire_logger.py`: 로깅 시스템 테스트
- `test_real_time_fire_detector.py`: 실시간 감지기 테스트

## 문제 해결

### 일반적인 문제

1. **웹캠 접근 실패**
   - 다른 카메라 인덱스 시도
   - 카메라 권한 확인

2. **YOLO 모델 로드 실패**
   - 모델 파일 경로 확인
   - 기본 모델 자동 다운로드

3. **성능 문제**
   - 해상도 낮추기
   - 감지 모드 변경

4. **영상 종료 후 계속 감지 중**
   - 자동으로 영상 종료 시 감지 중지 처리됨

### 로그 확인

```bash
# 로그 파일 위치
logs/fire_detection.log
logs/fire_detection_YYYYMMDD_HHMMSS.csv
```

## 프로젝트 구조

```
src/week4/
├── main.py                    # 메인 진입점
├── fire_detector_app.py       # GUI 애플리케이션
├── fire_detection_opencv.py   # OpenCV 감지 모듈
├── fire_detection_yolo.py     # YOLO 감지 모듈
├── real_time_fire_detector.py # 실시간 감지기
├── fire_logger.py            # 로깅 모듈
├── config.py                 # 설정 파일
├── requirements.txt          # 필수 패키지
├── test/                     # Unit Test 폴더
│   ├── test_fire_detection_opencv.py
│   ├── test_fire_detection_yolo.py
│   ├── test_fire_logger.py
│   └── test_real_time_fire_detector.py
├── logs/                     # 로그 파일 폴더
├── outputs/                  # 결과 영상 저장 폴더
└── README.md                # 프로젝트 문서
```

## 주요 특징

- **사용자 친화적 GUI**: 직관적인 인터페이스
- **다양한 입력 소스**: 웹캠, 비디오 파일
- **실시간 처리**: 실시간 화재 감지 및 시각화
- **자동 영상 종료 처리**: 영상이 끝나면 자동으로 감지 중지
- **결과 영상 저장**: bounding box가 포함된 감지 결과를 MP4로 저장
- **유연한 설정**: 감지 모드, 신뢰도 등 실시간 조정 가능
- **완전한 Unit Test**: 모든 모듈에 대한 테스트 제공
- **로깅 시스템**: CSV 형식으로 감지 결과 기록
