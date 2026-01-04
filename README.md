# 📘 CMT-CVAI 

## 📂 프로젝트 구조
```
CMT-CVAI/
├─ inputs/ # 실험용 입력 데이터
│ ├─ preprocessed_samples/ # 전처리된 샘플 이미지
│ └─ yolo_detection_samples/ # fire 객체 탐지 테스트 이미지
├─ outputs/ # 결과 저장 (Fire Detection, Depth Map 등)
├─ src/ # 소스 코드
│ ├─ week1/ # 1주차: 이미지 전처리
│ ├─ week2/ # 2주차: 2D → 3D 변환, pytest Unittest
│ ├─ week3/ # 3주차: YOLOv8 객체 탐지
│ └─ week4/ # 4주차: 실시간 화재 탐지 시스템
├─ venv2/ # 가상환경
└─ README.md
```
---

## 주차별 학습 내용

### 🔹 Week 1: 이미지 전처리 (Preprocessing)
- **파일**  
  - `image_preprocessing.py`: OpenCV를 활용한 기본 이미지 전처리 파이프라인  
  - `image_red_mask.py`: 특정 색상(빨강) 영역 추출 및 마스크 처리  

- **성과**  
  - 입력 이미지에서 원하는 색상/패턴 추출 및 전처리 자동화  
  - 함수 기반 모듈화로 재사용성 향상  
  - pathlib 기반 경로 처리 및 강화된 에러 핸들링  

---

### 🔹 Week 2: 2D → 3D 변환 및 테스트
- **파일**  
  - `processing_3d.py`, `processing_3d_hard.py`: Depth Map을 기반으로 3D 포인트 클라우드 생성  
  - `test_processing_3d.py`, `simple_unittest.py`: Unit test 코드 작성  

- **성과**  
  - OpenCV & NumPy를 사용한 3D 변환 파이프라인 구축  
  - Unit test로 코드 신뢰성 확보  
  - 타입 힌트 추가 및 함수 모듈화로 코드 품질 개선  

---

### 🔹 Week 3: YOLOv8 객체 탐지
- **파일**  
  - `train_yolo.py`: YOLOv8 모델 학습 (커맨드라인 인자 지원)  
  - `evaluate_yolo.py`: Precision / Recall / mAP 시각화 평가 (플롯 자동 저장)  
  - `detection_yolov8.py`: 학습된 모델로 객체 탐지 수행 (배치 처리 지원)  

- **성과**  
  - 커스텀 데이터셋(`datasets/train`, `val`, `test`) 기반 학습  
  - 성능 지표(Figure_1.png) 확인  
  - 커맨드라인 인터페이스 추가로 유연한 실행 환경 제공  
  - 강화된 에러 핸들링 및 로깅 시스템 
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/3c0afd22-d41d-4f3a-ad44-2b71ddcabe66" />
 
  - 이미지/영상 객체 탐지 구현  

---

### 🔹 Week 4: 실시간 화재 탐지 시스템
- **파일**  
  - `main.py`: 메인 진입점 (GUI/콘솔 선택)  
  - `fire_detector_app.py`: Tkinter 기반 GUI 애플리케이션  
  - `fire_detection_opencv.py`: OpenCV HSV 색공간 기반 화재 감지  
  - `fire_detection_yolo.py`: YOLOv8 기반 화재 객체 탐지  
  - `real_time_fire_detector.py`: 하이브리드 실시간 감지 시스템  
  - `fire_logger.py`: CSV 형식 로깅 시스템  
  - `config.py`: 설정 관리  

- **성과**  
  - **GUI 인터페이스**: 사용자 친화적인 실시간 화재 감지 애플리케이션  
  - **다중 감지 모드**: OpenCV, YOLO, 하이브리드 모드 지원  
  - **실시간 처리**: 웹캠/비디오 파일 실시간 화재 감지  
  - **결과 영상 저장**: bounding box가 포함된 감지 결과를 MP4로 저장  
  - **로깅 시스템**: CSV 형식으로 감지 결과 및 통계 기록  
  - **Unit Test**: 모든 모듈에 대한 완전한 테스트 커버리지  
  - **자동 영상 종료 처리**: 영상 재생 완료 시 자동 감지 중지  
  - **성능 최적화**: 프레임 스킵 및 리소스 관리 개선  
  - **코드 품질**: 타입 힌트, 에러 핸들링, 경로 처리 개선  

---

## ⚙️ 실행 환경

### Week 1-2 (Python 3.13)
- Python 3.13+
- OpenCV
- Matplotlib, NumPy

### Week 3-4 (Python 3.11)
- Python 3.11
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Matplotlib, NumPy
- Tkinter (GUI)
- Pillow (이미지 처리)
- Pandas (데이터 처리)

> **가상환경**:  
> - Week 1-2: `venv2/` (Python 3.13)  
> - Week 3-4: `venv311/` (Python 3.11)  
> - Week 4 추가 패키지: `src/week4/requirements.txt` 참조

---

## 실행 방법

### 1. 가상환경 활성화

#### Week 1-2 (Python 3.13)
```bash
# Windows
venv2/Scripts/activate

# Linux / Mac
source venv2/bin/activate
```

#### Week 3-4 (Python 3.11)
```bash
# Windows
venv311/Scripts/activate

# Linux / Mac
source venv311/bin/activate
```

### 2. 주차별 실행

#### Week 1-2 (venv2 환경에서 실행)
```bash
# 1주차 전처리
python src/week1/image_preprocessing.py

# 2주차 3D 변환
python src/week2/processing_3d.py
```

#### Week 3-4 (venv311 환경에서 실행)
```bash
# 3주차 YOLO 학습 (기본 설정)
python src/week3/train_yolo.py

# 3주차 YOLO 학습 (커맨드라인 옵션)
python src/week3/train_yolo.py --epochs 50 --batch 32 --device cuda

# 3주차 YOLO 객체 탐지
python src/week3/detection_yolov8.py --model runs/detect/train/weights/best.pt --input inputs/yolo_detection_samples/

# 3주차 YOLO 성능 평가
python src/week3/evaluate_yolo.py --model runs/detect/train/weights/best.pt

# 4주차 실시간 화재 탐지 시스템
python src/week4/main.py
```

## 결과 예시

**Week 1**: `outputs/sample_output.jpg` - 이미지 전처리 결과

**Week 2**: `outputs/depth_map1.jpg`, `outputs/depth_map2.jpg` - 3D 변환 결과

**Week 3**: `src/week3/Figure_1.png` - YOLO 성능 평가 그래프

**Week 4**: 
- `src/week4/outputs/` - 실시간 화재 감지 결과 영상 (MP4)
- `src/week4/logs/` - 화재 감지 로그 파일 (CSV)
- GUI 애플리케이션으로 실시간 화재 감지 및 시각화

---

## 🔧 최근 개선 사항

### 코드 품질 향상
- ✅ **경로 처리 개선**: `pathlib.Path` 사용으로 플랫폼 독립적 경로 관리
- ✅ **에러 핸들링 강화**: try-except-finally 블록으로 안전한 리소스 관리
- ✅ **타입 힌트 추가**: 모든 함수에 타입 힌트 추가로 코드 가독성 및 IDE 지원 향상
- ✅ **로깅 시스템 통일**: `logging` 모듈로 일관된 로그 관리
- ✅ **성능 최적화**: 프레임 스킵, 메모리 관리, 불필요한 연산 최소화
- ✅ **코드 모듈화**: 함수 기반 구조로 재사용성 및 유지보수성 향상
- ✅ **입력 검증 강화**: None 체크, 좌표 범위 검증, 값 유효성 검사
- ✅ **커맨드라인 인터페이스**: Week 3 스크립트에 argparse 기반 CLI 추가
