import sys, os
import numpy as np
import pytest

# 프로젝트 루트(CMT-CVAI)를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.week2.processing_3d_hard import generate_depth_map, generate_point

def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gray, depth_map = generate_depth_map(image)
    assert gray.shape == (100, 100), "Gray shape가 틀렸습니다."
    assert depth_map.shape == image.shape, "Depth map shape 오류"
    assert isinstance(depth_map, np.ndarray), "Depth map 타입 오류"

def test_generate_point():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    gray, depth_map = generate_depth_map(image)
    points_3d = generate_point(gray, depth_map)
    assert points_3d.shape == (200, 200, 3), "Point shape 오류"
