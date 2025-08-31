### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np

img = cv2.imread('./inputs/sample.jpg')
# 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Depth Map 생성
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
cv2.imwrite("./outputs/depth_map2.jpg", depth_map)
# 3D 포인트 클라우드 변환
h, w = depth_map.shape[:2]
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = gray.astype(np.float32) # Depth 값을 Z 축으로 사용
# 3D 좌표 생성
points_3d = np.dstack((X, Y, Z))

# 결과 출력
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()