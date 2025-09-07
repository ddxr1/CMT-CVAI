### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np

def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return gray, depth_map

def generate_point(gray, depth_map):
    if gray is None:
        raise ValueError("입력된 GrayScale 이미지가 없습니다.")
    if len(gray.shape) != 2:
        raise ValueError("Grayscale 이미지가 아닙니다.")
    # 3D 포인트 클라우드 변환
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)  # Depth 값을 Z 축으로 사용
    # 3D 좌표 생성
    points_3d = np.dstack((X, Y, Z))
    return points_3d

if __name__ == "__main__":
    img = cv2.imread("../../inputs/sample.jpg")
    gray, depth_map = generate_depth_map(img)
    cv2.imwrite("../../outputs/depth_map2.jpg", depth_map)
    points_3d = generate_point(gray, depth_map)

    # 결과 출력
    cv2.imshow('Depth Map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()