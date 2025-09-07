import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import cv2
import numpy as np
from src.week2.processing_3d import generate_depth_map

def test_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)
    assert depth_map.shape == image.shape, "출력 크기가 입력 이미지와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."