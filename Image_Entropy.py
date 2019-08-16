import cv2
import numpy as np
import math
from Util import ero_dila

image = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190808\\0808170829.jpg', 0)
chaos = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190812140738.jpg', 0)
grid = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\grid.png', 0)


# _, thresh = cv2.threshold(grid, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图
# cv2.imshow("SS", thresh)
# cv2.waitKey(0)


def get_entropy(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图
    thresh = ero_dila(thresh, 5)
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0

    img = np.array(thresh)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if tmp[i] == 0:
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

    return res


print(get_entropy(image))
print(get_entropy(chaos))
print(get_entropy(grid))

cv2.imshow("Image", image)
cv2.imshow("Chaos", chaos)

cv2.waitKey(0)
