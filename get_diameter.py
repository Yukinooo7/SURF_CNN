import cv2
from Util import ero_dila


def get_all_diameters(img):
    for i in range(img.shape[0]):
        get_diameter(img, i)


def get_diameter(img, y_axis):
    dia_list = []
    diameter = 0
    flag = False
    line_flag = 0
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图
    img = ero_dila(img, 1)

    for i in range(img.shape[1]):
        if img[y_axis][i] < 10:
            flag = True
            line_flag = line_flag + 1
        else:
            flag = False
            line_flag = 0
            if diameter != 0:
                dia_list.append(diameter)
            diameter = 0

        if flag:
            diameter = diameter + 1

        if line_flag > 80:
            print("It is a line")

            break

    print(dia_list)


if __name__ == "__main__":
    image = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190813151052.jpg', 0)

    get_all_diameters(image)

    _, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图
    img = ero_dila(img, 1)

    cv2.imshow("Origin", image)

    cv2.imshow("IMG", img)
    cv2.waitKey(0)
