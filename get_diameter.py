import cv2
from Util import ero_dila, mean
from matplotlib import pyplot

image = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190813151052.jpg', 0)

_, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图
img = ero_dila(img, 1)


def get_diagram(x, y):
    # 生成图表
    pyplot.plot(x, y)
    # # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
    # pyplot.xlabel('year')
    # pyplot.ylabel('population')
    # pyplot.title('Population year correspondence')
    # 设置纵坐标刻度
    pyplot.yticks([0, 25, 50, 75, 90])
    # 设置填充选项：参数分别对应横坐标，纵坐标，纵坐标填充起始值，填充颜色（可以有更多选项）
    pyplot.fill_between(x, y, 0, color='green')
    # 显示图表
    pyplot.show()


def get_all_diameters(img):
    for i in range(1, img.shape[0], 15):
        dia_list = get_diameter(img, i)
        if len(dia_list) > 0:
            print(len(dia_list))


def get_diameter(input_img, y_axis):
    dia_list = []
    diameter = 0
    flag = False
    line_flag = 0
    _, input_img = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图
    input_img = ero_dila(input_img, 1)

    for i in range(input_img.shape[1]):
        if input_img[y_axis][i] < 10:
            flag = True
            line_flag = line_flag + 1
        else:
            flag = False
            line_flag = 0
            if diameter > 5:
                cv2.circle(img, (i, y_axis), 5, (0, 0, 0), 1)
                dia_list.append(diameter)
            diameter = 0

        if flag:
            diameter = diameter + 1

        if line_flag > 80:
            print("It is a line")

            break

    return dia_list


if __name__ == "__main__":
    get_all_diameters(img)

    cv2.imshow("Origin", image)

    cv2.imshow("IMG", img)
    cv2.waitKey(0)
