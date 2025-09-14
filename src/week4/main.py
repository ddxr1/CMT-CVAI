"""
Fire Detection System - Main Entry Point
화재 객체 탐지 시스템 메인 진입점
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """메인 함수"""
    print("🔥 화재 객체 탐지 시스템 v1.0")
    print("=" * 40)
    print("1. GUI 애플리케이션 실행")
    print("2. 종료")
    print("=" * 40)
    
    while True:
        try:
            choice = input("선택하세요 (1-2): ").strip()
            
            if choice == "1":
                print("\nGUI 애플리케이션을 시작합니다...")
                from fire_detector_app import main as app_main
                app_main()
                break
                
            elif choice == "2":
                print("프로그램을 종료합니다.")
                break
                
            else:
                print("잘못된 선택입니다. 1-2 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            continue

if __name__ == "__main__":
    main()