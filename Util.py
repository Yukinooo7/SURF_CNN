import cv2
import numpy as np
from matplotlib import pyplot as plt
from correlation import get_contours, mean
import math

line_id = 0


class Point(object):
    def __init__(self, xParam=0.0, yParam=0.0):
        self.x = float(xParam)
        self.y = float(yParam)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def distance(self, pt):
        xDiff = self.x - pt.x
        yDiff = self.y - pt.y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def sum(self, pt):
        newPt = Point()
        xNew = self.x + pt.x
        yNew = self.y + pt.y
        return Point(xNew, yNew)


def get_function(point1, point2):
    k = get_slope(point1, point2)

    if k == math.inf:
        b = math.inf
    else:
        b = float(point1.y) - float(point1.x) * float(k)

    return k, b


class Line(object):
    def __init__(self, point1, point2, id):
        k, b = get_function(point1, point2)
        self.id = id
        self.k = k
        self.b = b
        self.origin_point = point1
        self.origin_point2 = point2
        print('Set up id: ' + str(self.id))

    def is_online(self, point):
        y = point.x * self.k + self.b

        if math.fabs(y - self.y) < 1:
            return True
        else:
            return False

    def is_parall(self, line):
        return abs(float(self.k) - float(line.k)) < 1


def erode(img, ite_time):
    kernel = np.ones((5, 5), np.uint8)
    img_ero = cv2.erode(img, kernel, iterations=ite_time)

    return img_ero


def dilation(img, ite_time):
    kernel = np.ones((5, 5), np.uint8)
    img_dila = cv2.dilate(img, kernel, iterations=ite_time)

    return img_dila


def ero_dila(img, ite_time):
    img = erode(img, ite_time)

    return dilation(img, ite_time)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    blank = np.zeros((image.shape[0], image.shape[1]))
    blank[blank < 1] = 255
    edges = cv2.Canny(image, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=30, maxLineGap=100)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow("line_detect_possible_demo", blank)


def convexHull(contours, img):
    # 图像凸包的处理过程
    for i in range(1, len(contours)):
        cnt = contours[i]
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        if defects is None:
            continue
        else:
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(img, start, end, [0, 255, 0], 5)

    return img


# 删除面积过小的轮廓
def contour_filter(contours):
    new_contours = []

    for c in contours:
        if cv2.contourArea(c) > 100:
            new_contours.append(c)

    return new_contours


# 获取逼近轮廓
def approx(contours):
    approx_con = []
    for c in contours:
        approx_con.append(cv2.approxPolyDP(c, 5, True))

    return approx_con


# 获取空白画布
def get_blank(img):
    blank = np.zeros((img.shape[0], img.shape[1]))
    blank[blank < 1] = 255

    return blank


# get slop value of two points
def get_slope(point1, point2):
    if point1.x == point2.x:
        return math.inf
    else:

        return format((point1.y - point2.y) / (point1.x - point2.x), '.1f')


# return point objects of one contour
def get_contour_points(contour):
    points = []

    for i in range(len(contour)):
        point = Point(contour[i][0][0], contour[i][0][1])  # [[256  51]]
        points.append(point)

    return points


# return distance
def point2line_distance(point, line):
    A = -float(line.k)
    B = 1
    C = -float(line.b)

    d = abs(A * point.x + B * point.y + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))

    return d


# caculate all distances of parallel lines, return a minimum value
def get_real_distance(l, lines):
    min = 1000
    real_line = None
    for line in lines:
        if l.id == line.id:
            print("Same id")
            continue

        if (l.k == line.k) & (l.b == line.b):
            continue
        else:
            if l.is_parall(line):
                if point2line_distance(l.origin_point, line) < min:
                    print("k: " + str(l.k) + " b: " + str(l.b), end=' ')
                    print(" k: " + str(line.k) + " b: " + str(line.b))
                    min = point2line_distance(l.origin_point, line)
                    real_line = line

    return min, real_line


# remove values which are too small or too big
def data_filter(distances, max_distance):
    result = []

    for d in distances:
        if d > 999 or d < 5:
            continue
        else:
            result.append(d)

    return result


def draw_line(blank, point1, point2):
    x1 = point1.x
    x2 = point2.x
    y1 = point1.y
    y2 = point1.y
    cv2.line(blank, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 100))

    return blank


def get_middle_point(point1, point2):
    return Point((point1.x + point2.x) / 2, (point1.y + point2.y) / 2)


def get_max_area(contours):
    max = 0

    for c in contours:
        if cv2.contourArea(c) > max:
            max = cv2.contourArea(c)

    return max


if __name__ == '__main__':
    # Reading the input image
    # img = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190808\\0808172539.jpg', 0)
    # img = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\Scaffold\\0730103043.jpg', 0)
    # img = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190808\\0808170829.jpg', 0)

    img = cv2.imread('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\images\\20190813125845.jpg')
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # 获取灰度图

    thresh2 = ero_dila(thresh2, 2)  # 边缘平滑处理

    contours = contour_filter(get_contours(thresh2))  # 获取轮廓（过大或者过小的轮廓去除）
    max_area = get_max_area(contours)
    max_distance = math.sqrt(max_area)

    approx_con = approx(contours)  # 获取逼近轮廓，返回一个数组，里面每个元素包含着一个轮廓的点，一般是四个

    blank = get_blank(img)  # 空白画布

    lines = []  # 存储Line对象

    # 每个轮廓，假设是四个顶点，那就对应着四条边，这一步操作将根据每个轮廓返回边存储在lines数组里面
    for i in range(len(approx_con)):
        line_id += 1
        points = get_contour_points(approx_con[i])  # 将每个轮廓对象变成一个含有Point对象的数组
        for j in range(0, len(points)):
            if j == (len(points) - 1):
                line = Line(points[j], points[0], line_id)
                # lines.append(Line(points[j], points[0]))
                # print("Slope: " + str(get_slope(points[j], points[0])), end="---")
            else:
                line = Line(points[j], points[j + 1], line_id)
                # print("Slope: " + str(get_slope(points[j], points[j + 1])), end="---")

            lines.append(line)

    # 设置一个temp变量存储lines，防止访问冲突
    temp = lines
    distances = []

    new_blank = get_blank(img)

    # 取出一条line，挨个比对这个数组里面的所有边，如果平行而且不为同一条线段，那么计算距离
    # 由于一个图里面可能会有很多个平行的线段，所以取距离最小的线段作为真实距离
    for line in lines:
        min, real_line = get_real_distance(line, temp)
        if real_line is not None:
            middle_point = get_middle_point(real_line.origin_point, real_line.origin_point2)
            cv2.circle(blank, (int(middle_point.x), int(middle_point.y)), 10, (0, 0, 0), 10)
            cv2.circle(blank, (int(line.origin_point.x), int(line.origin_point.y)), 10, (0, 0, 0), 0)
        distances.append(min)

    # 删去过于大的和过小的距离
    distances = data_filter(distances, max_distance)

    distances.sort()

    for d in distances:
        print(d)

    print("Mean: " + str(mean(distances)))

    # for c in approx_con:
    #     point1 = Point(c[0][0][0], c[0][0][1])
    #     print(point1)

    cv2.polylines(new_blank, approx_con, True, (0, 255, 0), 2)
    black_white = get_blank(img)

    # for i in range(50):
    #     blank[200][i] = 0  # draw a reference object

    cv2.imshow('black_white', thresh2)
    cv2.imshow('blank', new_blank)
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    # cv2.imshow('Input', img)
    #
    # cv2.waitKey(0)
