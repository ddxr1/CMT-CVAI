"""
실시간 화재 감지 시스템
OpenCV와 YOLO를 결합한 하이브리드 화재 감지 시스템
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple, List
import logging

from fire_detection_opencv import FireDetectorOpenCV
from fire_detection_yolo import FireDetectorYOLO
from fire_logger import FireLogger

class RealTimeFireDetector:
    """실시간 화재 감지 시스템"""
    
    def __init__(self, 
                 use_opencv: bool = True,
                 use_yolo: bool = True,
                 hybrid_mode: bool = True,
                 confidence_threshold: float = 0.5,
                 log_dir: str = "logs"):
        """
        실시간 화재 감지기 초기화
        
        Args:
            use_opencv: OpenCV 기반 감지 사용 여부
            use_yolo: YOLO 기반 감지 사용 여부
            hybrid_mode: 하이브리드 모드 (두 방법 모두 사용)
            confidence_threshold: 신뢰도 임계값
            log_dir: 로그 저장 디렉토리
        """
        self.use_opencv = use_opencv
        self.use_yolo = use_yolo
        self.hybrid_mode = hybrid_mode
        self.confidence_threshold = confidence_threshold
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 감지기 초기화
        self.opencv_detector = None
        self.yolo_detector = None
        
        if use_opencv:
            self.opencv_detector = FireDetectorOpenCV(
                confidence_threshold=confidence_threshold
            )
        
        if use_yolo:
            self.yolo_detector = FireDetectorYOLO(
                confidence_threshold=confidence_threshold
            )
        
        # 로거 초기화
        self.fire_logger = FireLogger(log_dir=log_dir)
        
        # 상태 변수
        self.is_running = False
        self.detection_thread = None
        self.current_frame = None
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # 1초 쿨다운
        
        # 통계
        self.total_frames = 0
        self.fire_detections = 0
    
    def start_detection(self, 
                       video_source: int = 0, 
                       output_file: Optional[str] = None,
                       show_preview: bool = True):
        """
        실시간 화재 감지 시작
        
        Args:
            video_source: 비디오 소스 (0: 웹캠, 파일 경로: 동영상 파일)
            output_file: 출력 비디오 파일 경로
            show_preview: 미리보기 창 표시 여부
        """
        # 이미 실행 중이면 종료
        if self.is_running:
            self.logger.warning("이미 감지가 실행 중입니다.")
            return
        
        try:
            # 비디오 캡처 초기화
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise Exception(f"비디오 소스 열기 실패: {video_source}")
            
            # 비디오 설정 (웹캠인 경우만)
            if isinstance(video_source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 출력 비디오 설정
            self.video_writer = None
            if output_file:
                from pathlib import Path
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 코덱 사용 (더 호환성 좋음)
                fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not self.video_writer.isOpened():
                    self.logger.warning(f"출력 비디오 파일을 열 수 없습니다: {output_file}")
                    self.video_writer = None
                else:
                    self.logger.info(f"출력 비디오 저장: {output_file}")
            
            self.is_running = True
            self.logger.info("실시간 화재 감지 시작")
            
            # 메인 루프
            self._detection_loop(show_preview)
            
        except Exception as e:
            self.logger.error(f"감지 시작 실패: {e}")
            raise
        finally:
            self.stop_detection()
    
    def _detection_loop(self, show_preview: bool):
        """감지 메인 루프"""
        frame_skip = 0  # 프레임 스킵 카운터 (성능 최적화)
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("프레임 읽기 실패 또는 비디오 종료")
                    break
                
                # 프레임 스킵 (성능 최적화 - 매 3프레임마다 1프레임만 처리)
                frame_skip += 1
                if frame_skip % 3 == 0:  # 매 3프레임마다 처리
                    continue
                
                self.current_frame = frame
                self.total_frames += 1
                
                # 화재 감지
                fire_detected, detection_info = self._detect_fire(frame)
                
                # 결과 시각화
                result_frame = self._visualize_results(frame, fire_detected, detection_info)
                
                # 로그 기록
                self._log_detection(fire_detected, detection_info)
                
                # 출력 비디오 저장
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(result_frame)
                
                # 미리보기 표시
                if show_preview:
                    cv2.imshow("Real-time Fire Detection", result_frame)
                    
                    # 키 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("사용자에 의해 종료됨")
                        break
                    elif key == ord('s'):
                        self._save_current_frame()
                    elif key == ord('i'):
                        self._print_detection_info()
                    elif key == ord(' '):  # 스페이스바로 일시정지
                        cv2.waitKey(0)
                
                # FPS 제한 (약 30 FPS)
                time.sleep(0.033)
                
        except KeyboardInterrupt:
            self.logger.info("키보드 인터럽트로 종료")
        except Exception as e:
            self.logger.error(f"감지 루프 중 오류 발생: {e}")
        finally:
            if show_preview:
                cv2.destroyAllWindows()
    
    def _detect_fire(self, frame: np.ndarray) -> Tuple[bool, dict]:
        """
        화재 감지 (하이브리드 모드)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (화재 검출 여부, 감지 정보)
        """
        detection_info = {
            "opencv": {"detected": False, "regions": [], "confidence": 0.0},
            "yolo": {"detected": False, "detections": [], "confidence": 0.0},
            "hybrid": {"detected": False, "confidence": 0.0}
        }
        
        # OpenCV 기반 감지
        if self.opencv_detector:
            opencv_detected, opencv_regions = self.opencv_detector.detect_fire(frame)
            if opencv_regions:
                max_confidence = max([region[4] for region in opencv_regions])
                detection_info["opencv"] = {
                    "detected": opencv_detected,
                    "regions": opencv_regions,
                    "confidence": max_confidence
                }
        
        # YOLO 기반 감지
        if self.yolo_detector:
            yolo_detected, yolo_detections = self.yolo_detector.detect_fire(frame)
            if yolo_detections:
                max_confidence = max([det[4] for det in yolo_detections])
                detection_info["yolo"] = {
                    "detected": yolo_detected,
                    "detections": yolo_detections,
                    "confidence": max_confidence
                }
        
        # 하이브리드 판단
        if self.hybrid_mode and self.use_opencv and self.use_yolo:
            # 두 방법 모두에서 감지되거나, 하나라도 높은 신뢰도로 감지
            opencv_conf = detection_info["opencv"]["confidence"]
            yolo_conf = detection_info["yolo"]["confidence"]
            
            hybrid_detected = (
                (detection_info["opencv"]["detected"] and detection_info["yolo"]["detected"]) or
                (opencv_conf > self.confidence_threshold * 1.5) or
                (yolo_conf > self.confidence_threshold * 1.5)
            )
            
            detection_info["hybrid"] = {
                "detected": hybrid_detected,
                "confidence": max(opencv_conf, yolo_conf)
            }
        else:
            # 단일 방법 사용
            if self.use_opencv:
                detection_info["hybrid"] = detection_info["opencv"]
            elif self.use_yolo:
                detection_info["hybrid"] = detection_info["yolo"]
        
        return detection_info["hybrid"]["detected"], detection_info
    
    def _visualize_results(self, 
                          frame: np.ndarray, 
                          fire_detected: bool, 
                          detection_info: dict) -> np.ndarray:
        """
        감지 결과 시각화
        
        Args:
            frame: 원본 프레임
            fire_detected: 화재 검출 여부
            detection_info: 감지 정보
            
        Returns:
            시각화된 프레임
        """
        result_frame = frame.copy()
        
        # OpenCV 감지 결과 그리기
        if detection_info["opencv"]["detected"]:
            result_frame = self.opencv_detector.draw_detections(
                result_frame, detection_info["opencv"]["regions"]
            )
        
        # YOLO 감지 결과 그리기
        if detection_info["yolo"]["detected"]:
            result_frame = self.yolo_detector.draw_detections(
                result_frame, detection_info["yolo"]["detections"]
            )
        
        # 상태 정보 표시
        self._draw_status_info(result_frame, fire_detected, detection_info)
        
        return result_frame
    
    def _draw_status_info(self, 
                         frame: np.ndarray, 
                         fire_detected: bool, 
                         detection_info: dict):
        """상태 정보를 프레임에 그리기"""
        height, width = frame.shape[:2]
        
        # 배경 사각형
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # 상태 텍스트
        status_text = "FIRE DETECTED!" if fire_detected else "Normal"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        
        cv2.putText(frame, status_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 통계 정보
        cv2.putText(frame, f"Frames: {self.total_frames}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Detections: {self.fire_detections}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 신뢰도 정보
        if detection_info["opencv"]["detected"]:
            conf = detection_info["opencv"]["confidence"]
            cv2.putText(frame, f"OpenCV: {conf:.2f}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if detection_info["yolo"]["detected"]:
            conf = detection_info["yolo"]["confidence"]
            cv2.putText(frame, f"YOLO: {conf:.2f}", (200, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _log_detection(self, fire_detected: bool, detection_info: dict):
        """감지 결과 로그 기록"""
        current_time = time.time()
        
        # 쿨다운 체크
        if fire_detected and (current_time - self.last_detection_time) < self.detection_cooldown:
            return
        
        if fire_detected:
            self.fire_detections += 1
            self.last_detection_time = current_time
        
        # 로그 기록
        method = "Hybrid" if self.hybrid_mode else ("OpenCV" if self.use_opencv else "YOLO")
        confidence = detection_info["hybrid"]["confidence"]
        
        # 위치 정보 추출
        location = {}
        if detection_info["opencv"]["regions"]:
            region = detection_info["opencv"]["regions"][0]  # 첫 번째 영역
            location = {"x": region[0], "y": region[1], "width": region[2], "height": region[3]}
        elif detection_info["yolo"]["detections"]:
            det = detection_info["yolo"]["detections"][0]  # 첫 번째 탐지
            location = {"x": det[0], "y": det[1], "width": det[2]-det[0], "height": det[3]-det[1]}
        
        # 프레임 크기 정보
        frame_size = (0, 0)
        if self.current_frame is not None:
            frame_size = (self.current_frame.shape[1], self.current_frame.shape[0])
        
        self.fire_logger.log_fire_detection(
            detection_method=method,
            fire_detected=fire_detected,
            confidence=confidence,
            location=location,
            frame_size=frame_size,
            additional_info=detection_info
        )
    
    def _save_current_frame(self):
        """현재 프레임 저장"""
        if self.current_frame is not None:
            from pathlib import Path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 출력 디렉토리 생성
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = output_dir / f"fire_detection_{timestamp}.jpg"
            if cv2.imwrite(str(filename), self.current_frame):
                self.logger.info(f"프레임 저장: {filename}")
            else:
                self.logger.error(f"프레임 저장 실패: {filename}")
        else:
            self.logger.warning("저장할 프레임이 없습니다.")
    
    def _print_detection_info(self):
        """감지 정보 출력"""
        stats = self.fire_logger.get_fire_statistics(hours=1)
        self.logger.info(f"감지 통계 (최근 1시간): {stats}")
    
    def stop_detection(self):
        """감지 중지 및 리소스 해제"""
        self.is_running = False
        
        try:
            # 비디오 캡처 해제
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
                self.logger.debug("비디오 캡처 해제 완료")
            
            # 비디오 라이터 해제
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                if self.video_writer.isOpened():
                    self.video_writer.release()
                self.video_writer = None
                self.logger.debug("비디오 라이터 해제 완료")
            
            # OpenCV 창 닫기
            cv2.destroyAllWindows()
            
            # 통계 로그 출력
            if self.total_frames > 0:
                detection_rate = (self.fire_detections / self.total_frames) * 100
                self.logger.info(f"화재 감지 중지 - 총 프레임: {self.total_frames}, "
                               f"화재 감지: {self.fire_detections}, "
                               f"감지율: {detection_rate:.2f}%")
            else:
                self.logger.info("화재 감지 중지")
                
        except Exception as e:
            self.logger.error(f"리소스 해제 중 오류 발생: {e}")
    
    def get_statistics(self) -> dict:
        """감지 통계 반환"""
        return {
            "total_frames": self.total_frames,
            "fire_detections": self.fire_detections,
            "detection_rate": self.fire_detections / self.total_frames if self.total_frames > 0 else 0,
            "log_statistics": self.fire_logger.get_fire_statistics(hours=24)
        }
