import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_color_spaces(image_path, mask_path):
    # 读取原始图像和掩码图像
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 确保掩码是二值的
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # 提取花萼部分和背景部分的像素
    flower_parts = image[mask == 255]
    background = image[mask == 0]
    background = background[::50]  # 每隔三个像素取一个

    # 转换到HSV和LAB颜色空间
    flower_parts_hsv = cv2.cvtColor(flower_parts.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    flower_parts_lab = cv2.cvtColor(flower_parts.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    background_hsv = cv2.cvtColor(background.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    background_lab = cv2.cvtColor(background.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    # 创建RGB空间的3D图
    fig_rgb = plt.figure()
    ax_rgb = fig_rgb.add_subplot(111, projection='3d')
    ax_rgb.scatter(flower_parts[:, 2], flower_parts[:, 1], flower_parts[:, 0], c='r', label='Flower Parts',alpha=0.01)
    ax_rgb.scatter(background[:, 2], background[:, 1], background[:, 0], c='b', label='Background',alpha=0.01)
    ax_rgb.set_title('RGB Color Space')
    ax_rgb.set_xlabel('Red')
    ax_rgb.set_ylabel('Green')
    ax_rgb.set_zlabel('Blue')
    ax_rgb.legend()
    plt.show()

    # 创建HSV空间的3D图
    fig_hsv = plt.figure()
    ax_hsv = fig_hsv.add_subplot(111, projection='3d')
    ax_hsv.scatter(flower_parts_hsv[:, 0], flower_parts_hsv[:, 1], flower_parts_hsv[:, 2], c='r', label='Flower Parts',alpha=0.01)
    ax_hsv.scatter(background_hsv[:, 0], background_hsv[:, 1], background_hsv[:, 2], c='b', label='Background',alpha=0.01)
    ax_hsv.set_title('HSV Color Space')
    ax_hsv.set_xlabel('Hue')
    ax_hsv.set_ylabel('Saturation')
    ax_hsv.set_zlabel('Value')
    ax_hsv.legend()
    plt.show()

    # 创建LAB空间的3D图
    fig_lab = plt.figure()
    ax_lab = fig_lab.add_subplot(111, projection='3d')
    ax_lab.scatter(flower_parts_lab[:, 0], flower_parts_lab[:, 1], flower_parts_lab[:, 2], c='r', label='Flower Parts',alpha=0.01)
    ax_lab.scatter(background_lab[:, 0], background_lab[:, 1], background_lab[:, 2], c='b', label='Background',alpha=0.01)
    ax_hsv.set_title('LAB Color Space')
    ax_hsv.set_xlabel('L')
    ax_hsv.set_ylabel('A')
    ax_hsv.set_zlabel('B')
    ax_hsv.legend()
    plt.show()


# 调用函数，确保替换下面的路径为你的图像路径
plot_color_spaces('/Users/xs/PycharmProjects/super-tomato/datasets_green/train/img/2.bmp',
                  '/Users/xs/PycharmProjects/super-tomato/datasets_green/train/label/2.png')
