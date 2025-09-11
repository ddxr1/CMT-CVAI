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
│ └─ week3/ # 3주차: YOLOv8 객체 탐지
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

---

### 🔹 Week 2: 2D → 3D 변환 및 테스트
- **파일**  
  - `processing_3d.py`, `processing_3d_hard.py`: Depth Map을 기반으로 3D 포인트 클라우드 생성  
  - `test_processing_3d.py`, `simple_unittest.py`: Unit test 코드 작성  

- **성과**  
  - OpenCV & NumPy를 사용한 3D 변환 파이프라인 구축  
  - Unit test로 코드 신뢰성 확보  

---

### 🔹 Week 3: YOLOv8 객체 탐지
- **파일**  
  - `train_yolo.py`: YOLOv8 모델 학습  
  - `evaluate_yolo.py`: Precision / Recall / mAP 시각화 평가  
  - `detection_yolov8.py`: 학습된 모델로 객체 탐지 수행  

- **성과**  
  - 커스텀 데이터셋(`datasets/train`, `val`, `test`) 기반 학습  
  - 성능 지표(Figure_1.png) 확인 
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/3c0afd22-d41d-4f3a-ad44-2b71ddcabe66" />
 
  - 이미지/영상 객체 탐지 구현  

---

## ⚙️ 실행 환경
- Python 3.13+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Matplotlib, NumPy

> 가상환경: `venv2/`

---

## 실행 방법

### 1. 가상환경 활성화
```bash
# Windows
venv2/Scripts/activate

# Linux / Mac
source venv2/bin/activate
```

# 1주차 전처리
python src/week1/image_preprocessing.py

# 2주차 3D 변환
python src/week2/processing_3d.py

# 3주차 YOLO 학습
python src/week3/train_yolo.py

## 결과 예시

outputs/depth_map1.jpg, outputs/depth_map2.jpg: 3D 변환 결과

outputs/sample_output.jpg: 이미지 전처리 결과

week3/Figure_1.png: YOLO 성능 평가 그래프
