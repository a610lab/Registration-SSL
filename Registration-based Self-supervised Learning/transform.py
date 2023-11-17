import numpy as np
import cv2
import numpy as np
import random
import math

class trans():
    def __init__(self, img, pi):
        width, height = img.shape[:2]
        pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
        pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T

        self.img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T
        self.cita = compute_G(self.img_coordinate, pi, height, width)
        self.pi = pi
        self.W, self.A, self.Z = pre_compute_waz(self.pi, height, width, self.img_coordinate)
        self.height = height
        self.width = width

    def deformation(self, img, qi):
        qi = self.pi * 2 - qi
        mapxy = np.swapaxes(
            np.float32(compute_fv(qi, self.W, self.A, self.Z, self.height, self.width, self.cita, self.img_coordinate)),
            0, 1)
        img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP, interpolation=cv2.INTER_LINEAR)

        return img


def pre_compute_waz(pi, height, width, img_coordinate):
    '''

    :param pi:
    :param height:
    :param
    :param img_coordinate: 坐标信息矩阵
    :return:
    '''

    # height*width*控制点个数
    wi = np.reciprocal(
        np.power(np.linalg.norm(np.subtract(pi, img_coordinate.reshape(height, width, 1, 2)) + 0.000000001, axis=3), 2))

    # height*width*2
    pstar = np.divide(np.matmul(wi, pi), np.sum(wi, axis=2).reshape(height, width, 1))

    # height*width*控制点个数*2
    phat = np.subtract(pi, pstar.reshape(height, width, 1, 2))

    z1 = np.subtract(img_coordinate, pstar)
    z2 = np.repeat(np.swapaxes(np.array([z1[:, :, 1], -z1[:, :, 0]]), 1, 2).T.reshape(height, width, 1, 2, 1),
                   [pi.shape[0]], axis=2)

    # height*width*控制点个数*2*1
    z1 = np.repeat(z1.reshape(height, width, 1, 2, 1), [pi.shape[0]], axis=2)

    # height*width*控制点个数*1*2
    s1 = phat.reshape(height, width, pi.shape[0], 1, 2)
    s2 = np.concatenate((s1[:, :, :, :, 1], -s1[:, :, :, :, 0]), axis=3).reshape(height, width, pi.shape[0], 1, 2)

    a = np.matmul(s1, z1)
    b = np.matmul(s1, z2)
    c = np.matmul(s2, z1)
    d = np.matmul(s2, z2)

    # 重构wi形状
    ws = np.repeat(wi.reshape(height, width, pi.shape[0], 1), [4], axis=3)

    # height*width*控制点个数*2*2
    A = (ws * np.concatenate((a, b, c, d), axis=3).reshape(height, width, pi.shape[0], 4)).reshape(height, width,
                                                                                                   pi.shape[0], 2, 2)

    return wi, A, z1


def compute_fv(qi, W, A, Z, height, width, cita, img_coordinate):
    '''
    :param
    qi:
    :param
    W:
    :param
    A:
    :param
    Z:
    :param
    height:
    :param

    :param
    cita: 衰减系数，减少局部变形对整体的影响
    :param
    img_coordinate:
    :return:
    '''

    qstar = np.divide(np.matmul(W, qi), np.sum(W, axis=2).reshape(height, width, 1))

    qhat = np.subtract(qi, qstar.reshape(height, width, 1, 2)).reshape(height, width, qi.shape[0], 1, 2)

    fv_ = np.sum(np.matmul(qhat, A), axis=2)

    fv = np.linalg.norm(Z[:, :, 0, :, :], axis=2) / (np.linalg.norm(fv_, axis=3) + 0.0000000001) * fv_[:, :, 0,
                                                                                                   :] + qstar

    fv = (fv - img_coordinate) * cita.reshape(height, width, 1) + img_coordinate

    return fv


def compute_G(img_coordinate, pi, height, width, thre=0.3):
    '''
    衰减系数计算
    :param img_coordinate:
    :param pi:
    :param height:
    :param
    :param thre: 影响系数，数值越大对控制区域外影响越大，反之亦然，取值范围0到无穷大
    :return:
    '''
    max = np.max(pi, 0)
    min = np.min(pi, 0)

    length = np.max(max - min)

    # 计算控制区域中心
    # p_ = (max + min) // 2
    p_ = np.sum(pi, axis=0) // pi.shape[0]

    # 计算控制区域
    minx, miny = min - length
    maxx, maxy = max + length
    minx = minx if minx > 0 else 0
    miny = miny if miny > 0 else 0
    maxx = maxx if maxx < height else height
    maxy = maxy if maxy < width else width

    k1 = (p_ - [0, 0])[1] / (p_ - [0, 0])[0]
    k2 = (p_ - [height, 0])[1] / (p_ - [height, 0])[0]
    k4 = (p_ - [0, width])[1] / (p_ - [0, width])[0]
    k3 = (p_ - [height, width])[1] / (p_ - [height, width])[0]
    k = (np.subtract(p_, img_coordinate)[:, :, 1] / (
                np.subtract(p_, img_coordinate)[:, :, 0] + 0.000000000001)).reshape(height, width, 1)
    k = np.concatenate((img_coordinate, k), axis=2)

    k[:, :p_[1], 0][(k[:, :p_[1], 2] > k1) | (k[:, :p_[1], 2] < k2)] = \
    (np.subtract(p_[1], k[:, :, 1]) / p_[1]).reshape(height, width, 1)[:, :p_[1], 0][
        (k[:, :p_[1], 2] > k1) | (k[:, :p_[1], 2] < k2)]
    k[:, p_[1]:, 0][(k[:, p_[1]:, 2] > k3) | (k[:, p_[1]:, 2] < k4)] = \
    (np.subtract(k[:, :, 1], p_[1]) / (width - p_[1])).reshape(height, width, 1)[:, p_[1]:, 0][
        (k[:, p_[1]:, 2] > k3) | (k[:, p_[1]:, 2] < k4)]
    k[:p_[0], :, 0][(k1 >= k[:p_[0], :, 2]) & (k[:p_[0], :, 2] >= k4)] = \
    (np.subtract(p_[0], k[:, :, 0]) / p_[0]).reshape(height, width, 1)[:p_[0], :, 0][
        (k1 >= k[:p_[0], :, 2]) & (k[:p_[0], :, 2] >= k4)]
    k[p_[0]:, :, 0][(k3 >= k[p_[0]:, :, 2]) & (k[p_[0]:, :, 2] >= k2)] = \
    (np.subtract(k[:, :, 0], p_[0]) / (height - p_[0])).reshape(height, width, 1)[p_[0]:, :, 0][
        (k3 >= k[p_[0]:, :, 2]) & (k[p_[0]:, :, 2] >= k2)]

    cita = np.exp(-np.power(k[:, :, 0] / thre, 2))
    cita[minx:maxx, miny:maxy] = 1
    # 如果不需要局部变形，可以把cita的值全置为1
    # cita = 1

    return cita
a = np.array([(np.random.randint(10,90),np.random.randint(20,60)) for _ in range (20)])
c = []
def generate_points(center_x, center_y):
        radius2 = np.random.randint(-5, 5)
        radius1 = np.random.randint(-3, 3)
        x = center_x + radius1
        y = center_y + radius2
        points = (x, y)
        return points
for number in a:
    x, y = number
    c.append(generate_points(x, y))
b = np.array(c)


def tran(img, pi=a, qi=b):
    ddd = trans(img, pi)
    img = ddd.deformation(img, qi)
    return img

if __name__ == "__main__":
    img=cv2.imread(r'C:\Users\DJ\Desktop\lena.jpg',cv2.IMREAD_COLOR)
    img0 = img

    """
    pi:初始点位，这里定位双眼，鼻尖，双腮边缘共5个点
    qi:目标点位，保持双眼，鼻尖不动，双腮边缘向内侧移动20个像素点，呈现效果为瘦脸
    """
# a = np.array([200, 200, 315, 200, 260, 260, 160, 326, 356, 326]).reshape(-1, 2)  # -1代表不设限制，由程序自己计算是多少
# b = np.array([220, 220, 315, 200, 260, 260, 180, 326, 336, 326]).reshape(-1, 2)

    # 用绿点标出出初始点，观察是否定位准确
    # for i in range(len(pi)):
    #     cv2.circle(img0, (pi[i][0], pi[i][1]), 2, (0,255,0))
    # cv2.namedWindow("sour", 0)
    # cv2.resizeWindow("sour", 640, 480)
    # cv2.imshow('sour', img0)
    # cv2.waitKey(0)

    # 开始图像变形

    img =tran(img)

    # 展示源图
    cv2.namedWindow("sour", 0)  # 显示原图
    cv2.resizeWindow("sour",256, 256)
    cv2.imshow('sour', img0)

    # 展示效果图
    cv2.namedWindow("bianxing", 0)  # 显示效果图
    cv2.resizeWindow("bianxing",256,256)
    cv2.imshow('bianxing', img)

    cv2.waitKey(0)