### 기본적인 Depth Map 생성 코드 (Opencv 활용)
import cv2
import numpy as np

img = cv2.imread("./inputs/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
cv2.imwrite("./outputs/depth_map1.jpg", depth_map)
cv2.imshow("Origin", img)
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()