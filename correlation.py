import os
import cv2
import numpy as np
import math

sample = './20190708_pictures/07081018.jpg'


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def get_contours(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图处理
    ret, thresh = cv2.threshold(img, 100, 255, 0)  # 二值化处理，参数需要调整
    '''
    cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])

    第一个参数是寻找轮廓的图像；

    第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
        cv2.RETR_EXTERNAL表示只检测外轮廓
        cv2.RETR_LIST检测的轮廓不建立等级关系
        cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        cv2.RETR_TREE建立一个等级树结构的轮廓。

    第三个参数method为轮廓的近似办法
        cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    '''

    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)
    '''
    返回值：
    (OpenCV 3版本会返回3个值)

    1. Binary: 不知道
    2. Contours：Numpy列表，存着所有的contours，需要用循环读取所有的contour
    3. Hierarchy：轮廓的层次结构，基本不用
    '''

    return contours


def get_min_matchshapes(contour):
    square_contours = get_contours(cv2.imread('E:\\Study\\2019-Summer\\SURF\\python\\squares.png'))
    min_matchpoint = 1

    for square in square_contours:
        temp = cv2.matchShapes(contour, square, 1, 0.0)
        if temp < min_matchpoint:
            min_matchpoint = temp

    return min_matchpoint


def get_scaffold_data(url):
    img = cv2.imread(url)
    contours = get_contours(img)
    area_perimeter = dict()
    img_area = img.shape[0] * img.shape[1]

    # 参数一：需要排序的数组
    # 参数二：排序规则，根据x的相对应的面积进行排序，输入x，返回x的面积
    # 参数三：顺序还是反序
    # sortedContors = sorted(contours, key=cv2.contourArea, reverse=True)

    sortedContors = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)

    # (image to draw, contours(一个NP数组), 第几个contour, color, thickness)
    # 第一个轮廓必须去除，因为最大的那个轮廓就是把整张图片连起来的那个
    for i in range(1, len(contours)):
        area = cv2.contourArea(sortedContors[i])
        perimeter = cv2.arcLength(sortedContors[i], True)
        if area > 100:
            cv2.drawContours(img, sortedContors, i, (0, 255 - 5 * i, 0), 5)
            area_perimeter[area] = perimeter

    # count = 0
    #
    # # 图像凸包的处理过程
    # for i in range(1, len(sortedContors)):
    #     cnt = sortedContors[i]
    #     hull = cv2.convexHull(cnt, returnPoints=False)
    #     defects = cv2.convexityDefects(cnt, hull)
    #
    #     if defects is None:
    #         continue
    #     else:
    #         count += 1
    #         for j in range(defects.shape[0]):
    #             s, e, f, d = defects[j, 0]
    #             start = tuple(cnt[s][0])
    #             end = tuple(cnt[e][0])
    #             cv2.line(img, start, end, [0, 255, 0], 5)

    return area_perimeter.keys()


def get_urls(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    urls = []

    for file in files:
        img_url = path + '\\' + file
        urls.append(img_url)

    return urls


def mean(x):
    return sum(x) / len(x)


def perimeter_area_ratio(perimeter, area):
    return perimeter / area


# 计算每一项数据与均值的差
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


# 辅助计算函数 dot product 、sum_of_squares
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


# 方差
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)


# 标准差
def standard_deviation(x):
    return math.sqrt(variance(x))


# 协方差
def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


# 相关系数
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0


if __name__ == '__main__':

    stdev_set = []
    mean_set = []
    variance_set = []
    path = "E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\Scaffold"  # 文件夹目录

    for url in get_urls(path):
        areas = get_scaffold_data(url)
        if len(areas) > 2:
            area_stdev = standard_deviation(areas)
            area_variance = variance(areas)
            area_mean = mean(areas)

            stdev_set.append(area_stdev)
            mean_set.append(area_mean)
            variance_set.append(area_variance)

    correlation = correlation(variance_set, mean_set) # 0.54, 300 samples

    print("correlation: " + str(correlation))
