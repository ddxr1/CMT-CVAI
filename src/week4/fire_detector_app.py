"""
Fire Detection Software Prototype
화재 객체 탐지 소프트웨어 프로토타입
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
from pathlib import Path

# 감지 모듈들 import
from fire_detection_opencv import FireDetectorOpenCV
from fire_detection_yolo import FireDetectorYOLO
from fire_logger import FireLogger

class FireDetectorApp:
    """화재 감지 소프트웨어 메인 애플리케이션"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("화재 객체 탐지 시스템 v1.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 상태 변수
        self.is_detecting = False
        self.current_video = None
        self.cap = None
        self.detection_thread = None
        
        # 영상 저장 관련
        self.video_writer = None
        self.output_video_path = None
        self.save_video = False
        
        # 감지기들
        self.opencv_detector = None
        self.yolo_detector = None
        self.logger = FireLogger(log_dir="logs", log_format="csv")  # CSV만 생성
        
        # 통계
        self.total_frames = 0
        self.fire_detections = 0
        
        self.setup_ui()
        self.setup_detectors()
    
    def setup_ui(self):
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="🔥 화재 객체 탐지 시스템", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 왼쪽 패널 - 컨트롤
        control_frame = ttk.LabelFrame(main_frame, text="제어 패널", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 비디오 선택
        ttk.Label(control_frame, text="비디오 소스:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_source_var = tk.StringVar(value="웹캠")
        video_combo = ttk.Combobox(control_frame, textvariable=self.video_source_var, 
                                  values=["웹캠", "파일 선택"], state="readonly", width=15)
        video_combo.grid(row=0, column=1, pady=5, padx=(5, 0))
        video_combo.bind('<<ComboboxSelected>>', self.on_video_source_change)
        
        # 파일 선택 버튼
        self.file_button = ttk.Button(control_frame, text="파일 선택", 
                                     command=self.select_video_file, state="disabled")
        self.file_button.grid(row=0, column=2, pady=5, padx=(5, 0))
        
        # 감지 모드
        ttk.Label(control_frame, text="감지 모드:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.detection_mode_var = tk.StringVar(value="하이브리드")
        mode_combo = ttk.Combobox(control_frame, textvariable=self.detection_mode_var,
                                 values=["OpenCV", "YOLO", "하이브리드"], state="readonly", width=15)
        mode_combo.grid(row=1, column=1, pady=5, padx=(5, 0))
        
        # 신뢰도 임계값
        ttk.Label(control_frame, text="신뢰도 임계값:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # 영상 저장 옵션
        self.save_video_var = tk.BooleanVar(value=False)
        save_checkbox = ttk.Checkbutton(control_frame, text="결과 영상 저장", 
                                       variable=self.save_video_var)
        save_checkbox.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5, padx=(5, 0))
        
        # 시작/중지 버튼
        self.start_button = ttk.Button(control_frame, text="감지 시작", 
                                      command=self.start_detection, style="Accent.TButton")
        self.start_button.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        
        self.stop_button = ttk.Button(control_frame, text="감지 중지", 
                                     command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=4, column=1, pady=10, padx=(5, 0), sticky=(tk.W, tk.E))
        
        # 통계 정보
        stats_frame = ttk.LabelFrame(control_frame, text="통계", padding="5")
        stats_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=30, font=('Consolas', 9))
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 오른쪽 패널 - 비디오 표시
        video_frame = ttk.LabelFrame(main_frame, text="비디오 화면", padding="10")
        video_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # 비디오 라벨
        self.video_label = ttk.Label(video_frame, text="비디오를 선택하고 감지를 시작하세요", 
                                    font=('Arial', 12), anchor='center')
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 상태바
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="준비됨")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 진행률 표시
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          mode='indeterminate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(20, 0))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        control_frame.columnconfigure(1, weight=1)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=1)
    
    def setup_detectors(self):
        """감지기 초기화"""
        try:
            self.opencv_detector = FireDetectorOpenCV(
                lower_hsv=(0, 50, 50),
                upper_hsv=(35, 255, 255),
                min_area=500,
                confidence_threshold=0.3
            )
            
            self.yolo_detector = FireDetectorYOLO(
                model_path="../week3/runs/detect/train/weights/best.pt",
                confidence_threshold=0.5
            )
            
            self.status_var.set("감지기 초기화 완료")
        except Exception as e:
            messagebox.showerror("오류", f"감지기 초기화 실패: {str(e)}")
            self.status_var.set("감지기 초기화 실패")
    
    def on_video_source_change(self, event):
        """비디오 소스 변경 시 호출"""
        if self.video_source_var.get() == "파일 선택":
            self.file_button.config(state="normal")
        else:
            self.file_button.config(state="disabled")
            self.current_video = None
    
    def select_video_file(self):
        """비디오 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="비디오 파일 선택",
            filetypes=[
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            self.current_video = file_path
            self.status_var.set(f"선택된 파일: {os.path.basename(file_path)}")
    
    def start_detection(self):
        """감지 시작"""
        if self.is_detecting:
            return
        
        # 비디오 소스 설정
        if self.video_source_var.get() == "웹캠":
            video_source = 0
        else:
            if not self.current_video:
                messagebox.showwarning("경고", "비디오 파일을 선택해주세요.")
                return
            video_source = self.current_video
        
        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            messagebox.showerror("오류", "비디오 소스를 열 수 없습니다.")
            return
        
        # 영상 저장 설정
        self.save_video = self.save_video_var.get()
        if self.save_video:
            self.setup_video_writer()
        
        # UI 상태 변경
        self.is_detecting = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_bar.start()
        self.status_var.set("감지 중...")
        
        # 감지 스레드 시작
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def setup_video_writer(self):
        """영상 저장을 위한 VideoWriter 설정"""
        try:
            # 출력 파일 경로 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_video_path = f"outputs/fire_detection_{timestamp}.mp4"
            
            # 출력 디렉토리 생성
            os.makedirs("outputs", exist_ok=True)
            
            # 비디오 속성 가져오기
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # VideoWriter 초기화
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_video_path, fourcc, fps, (width, height)
            )
            
            if not self.video_writer.isOpened():
                self.video_writer = None
                self.save_video = False
                messagebox.showwarning("경고", "영상 저장을 초기화할 수 없습니다.")
            else:
                print(f"영상 저장 시작: {self.output_video_path}")
                
        except Exception as e:
            print(f"영상 저장 초기화 오류: {e}")
            self.video_writer = None
            self.save_video = False
    
    def stop_detection(self):
        """감지 중지"""
        self.is_detecting = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 영상 저장 종료
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            if self.output_video_path:
                messagebox.showinfo("완료", f"영상이 저장되었습니다:\n{self.output_video_path}")
                self.output_video_path = None
        
        # UI 상태 변경
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_bar.stop()
        self.status_var.set("감지 중지됨")
        
        # 최종 통계 표시
        self.update_statistics()
    
    def on_video_end(self):
        """영상 종료 시 처리"""
        self.is_detecting = False
        self.status_var.set("영상 종료됨")
        
        # 진행 바 중지
        self.progress_bar.stop()
        
        # 영상 저장 종료
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            if self.output_video_path:
                messagebox.showinfo("완료", f"영상이 저장되었습니다:\n{self.output_video_path}")
                self.output_video_path = None
        
        # 버튼 상태 업데이트
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        # 비디오 캡처 해제
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 최종 통계 표시
        self.update_statistics()
        
        # 메시지 표시
        messagebox.showinfo("영상 종료", "영상 재생이 완료되었습니다.")
    
    def detection_loop(self):
        """감지 메인 루프"""
        while self.is_detecting and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                # 영상이 끝났을 때 UI 상태 업데이트
                self.root.after(0, self.on_video_end)
                break
            
            self.total_frames += 1
            
            # 현재 프레임 저장
            self.current_frame = frame
            
            # 화재 감지
            fire_detected, detection_info = self.detect_fire(frame)
            
            if fire_detected:
                self.fire_detections += 1
            
            # 로깅
            self.log_detection(fire_detected, detection_info)
            
            # 결과 시각화
            result_frame = self.visualize_results(frame, fire_detected, detection_info)
            
            # 영상 저장
            if self.save_video and self.video_writer:
                self.video_writer.write(result_frame)
            
            # UI 업데이트
            self.root.after(0, self.update_video_display, result_frame)
            self.root.after(0, self.update_statistics)
            
            # FPS 제한
            time.sleep(0.033)  # 약 30 FPS
    
    def detect_fire(self, frame):
        """화재 감지"""
        mode = self.detection_mode_var.get()
        confidence = self.confidence_var.get()
        
        detection_info = {
            "opencv": {"detected": False, "regions": [], "confidence": 0.0},
            "yolo": {"detected": False, "detections": [], "confidence": 0.0},
            "hybrid": {"detected": False, "confidence": 0.0}
        }
        
        # OpenCV 감지
        if mode in ["OpenCV", "하이브리드"] and self.opencv_detector:
            opencv_detected, opencv_regions = self.opencv_detector.detect_fire(frame)
            if opencv_regions:
                max_confidence = max([region[4] for region in opencv_regions])
                detection_info["opencv"] = {
                    "detected": opencv_detected,
                    "regions": opencv_regions,
                    "confidence": max_confidence
                }
        
        # YOLO 감지
        if mode in ["YOLO", "하이브리드"] and self.yolo_detector:
            yolo_detected, yolo_detections = self.yolo_detector.detect_fire(frame)
            if yolo_detections:
                max_confidence = max([det[4] for det in yolo_detections])
                detection_info["yolo"] = {
                    "detected": yolo_detected,
                    "detections": yolo_detections,
                    "confidence": max_confidence
                }
        
        # 하이브리드 판단
        if mode == "하이브리드":
            opencv_conf = detection_info["opencv"]["confidence"]
            yolo_conf = detection_info["yolo"]["confidence"]
            
            hybrid_detected = (
                (detection_info["opencv"]["detected"] and detection_info["yolo"]["detected"]) or
                (opencv_conf > confidence * 1.5) or
                (yolo_conf > confidence * 1.5)
            )
            
            detection_info["hybrid"] = {
                "detected": hybrid_detected,
                "confidence": max(opencv_conf, yolo_conf)
            }
        elif mode == "OpenCV":
            detection_info["hybrid"] = detection_info["opencv"]
        elif mode == "YOLO":
            detection_info["hybrid"] = detection_info["yolo"]
        
        return detection_info["hybrid"]["detected"], detection_info
    
    def log_detection(self, fire_detected, detection_info):
        """감지 결과 로깅"""
        try:
            mode = self.detection_mode_var.get()
            confidence = detection_info["hybrid"]["confidence"]
            
            # 위치 정보 추출
            location = {}
            if detection_info["opencv"]["regions"]:
                region = detection_info["opencv"]["regions"][0]
                location = {"x": region[0], "y": region[1], "width": region[2], "height": region[3]}
            elif detection_info["yolo"]["detections"]:
                det = detection_info["yolo"]["detections"][0]
                location = {"x": det[0], "y": det[1], "width": det[2]-det[0], "height": det[3]-det[1]}
            
            # 프레임 크기 (현재 프레임에서 가져오기)
            frame_size = (0, 0)
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                frame_size = (self.current_frame.shape[1], self.current_frame.shape[0])
            
            # 로그 기록
            self.logger.log_fire_detection(
                detection_method=mode,
                fire_detected=fire_detected,
                confidence=confidence,
                location=location,
                frame_size=frame_size,
                additional_info=detection_info
            )
        except Exception as e:
            print(f"로깅 오류: {e}")
    
    def visualize_results(self, frame, fire_detected, detection_info):
        """결과 시각화"""
        result_frame = frame.copy()
        
        # OpenCV 감지 결과 그리기
        if detection_info["opencv"]["detected"] and self.opencv_detector:
            result_frame = self.opencv_detector.draw_detections(
                result_frame, detection_info["opencv"]["regions"]
            )
        
        # YOLO 감지 결과 그리기
        if detection_info["yolo"]["detected"] and self.yolo_detector:
            result_frame = self.yolo_detector.draw_detections(
                result_frame, detection_info["yolo"]["detections"]
            )
        
        # 상태 정보 표시
        height, width = result_frame.shape[:2]
        
        # 배경 사각형
        cv2.rectangle(result_frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(result_frame, (10, 10), (400, 100), (255, 255, 255), 2)
        
        # 상태 텍스트
        status_text = "FIRE DETECTED!" if fire_detected else "Normal"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        
        cv2.putText(result_frame, status_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 통계 정보
        cv2.putText(result_frame, f"Frames: {self.total_frames}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_frame, f"Detections: {self.fire_detections}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def update_video_display(self, frame):
        """비디오 표시 업데이트"""
        # OpenCV BGR을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(frame_rgb)
        
        # 크기 조정 (최대 800x600)
        max_width, max_height = 800, 600
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # Tkinter용 이미지로 변환
        photo = ImageTk.PhotoImage(pil_image)
        
        # 라벨 업데이트
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # 참조 유지
    
    def update_statistics(self):
        """통계 업데이트"""
        detection_rate = (self.fire_detections / self.total_frames * 100) if self.total_frames > 0 else 0
        
        stats_text = f"""총 프레임: {self.total_frames}
화재 감지: {self.fire_detections}
감지율: {detection_rate:.1f}%

현재 모드: {self.detection_mode_var.get()}
신뢰도: {self.confidence_var.get():.2f}

상태: {'감지 중' if self.is_detecting else '대기 중'}"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

def main():
    """메인 함수"""
    root = tk.Tk()
    app = FireDetectorApp(root)
    
    # 창 닫기 이벤트 처리
    def on_closing():
        if app.is_detecting:
            app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
