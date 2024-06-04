# -*- coding: utf-8 -*-
# @Time    : 2024/6/4 21:34
# @Author  : GG
# @File    : classifer.py
# @Software: PyCharm


import cv2
import numpy as np

class Tomato:
    def __init__(self):
        ''' 初始化 Tomato 类。'''
        pass

    def extract_s_l(self, image):
        '''
        提取图像的 S 通道（饱和度）和 L 通道（亮度），并将两者相加。
        :param image: 输入的 BGR 图像
        :return: S 通道和 L 通道相加的结果
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        s_channel = hsv[:, :, 1]
        l_channel = lab[:, :, 0]
        result = cv2.add(s_channel, l_channel)
        return result

    def find_reflection(self, image, threshold=190):
        '''
        通过阈值处理识别图像中的反射区域。
        :param image: 输入的单通道图像
        :param threshold: 用于二值化的阈值
        :return: 二值化后的图像，高于阈值的部分为白色，其余为黑色
        '''
        _, reflection = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return reflection

    def otsu_threshold(self, image):
        '''
        使用 Otsu 大津法自动计算并应用阈值，进行图像的二值化处理。
        :param image: 输入的单通道图像
        :return: 二值化后的图像
        '''
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def extract_g_r(self, image):
        '''
        提取图像中的 G 通道（绿色），放大并减去 R 通道（红色）。
        :param image: 输入的 BGR 图像
        :return: G 通道乘以 1.5 后减去 R 通道的结果
        '''
        g_channel = image[:, :, 1]
        r_channel = image[:, :, 2]
        result = cv2.subtract(cv2.multiply(g_channel, 1.5), r_channel)
        return result

    def extract_r_b(self, image):
        '''
        提取图像中的 R 通道（红色）和 B 通道（蓝色），并进行相减。
        :param image: 输入的 BGR 图像
        :return: R 通道减去 B 通道的结果
        '''
        r_channel = image[:, :, 2]
        b_channel = image[:, :, 0]
        result = cv2.subtract(r_channel, b_channel)
        return result

    def extract_r_g(self, image):
        '''
        提取图像中的 R 通道（红色）和 G 通道（绿色），并进行相减。
        :param image: 输入的 BGR 图像
        :return: R 通道减去 G 通道的结果
        '''
        r_channel = image[:, :, 2]
        g_channel = image[:, :, 1]
        result = cv2.subtract(r_channel, g_channel)
        return result

    def threshold_segmentation(self, image, threshold, color=255):
        '''
        对图像进行阈值分割，高于阈值的部分设置为指定的颜色。
        :param image: 输入的单通道图像
        :param threshold: 阈值
        :param color: 设置的颜色值
        :return: 分割后的二值化图像
        '''
        _, result = cv2.threshold(image, threshold, color, cv2.THRESH_BINARY)
        return result

    def bitwise_operation(self, image1, image2, operation='and'):
        '''
        对两幅图像执行位运算（与或运算）。
        :param image1: 第一幅图像
        :param image2: 第二幅图像
        :param operation: 执行的操作类型（'and' 或 'or'）
        :return: 位运算后的结果
        '''
        if operation == 'and':
            result = cv2.bitwise_and(image1, image2)
        elif operation == 'or':
            result = cv2.bitwise_or(image1, image2)
        else:
            raise ValueError("operation must be 'and' or 'or'")
        return result

    def largest_connected_component(self, bin_img):
        '''
        提取二值图像中的最大连通区域。
        :param bin_img: 输入的二值图像
        :return: 只包含最大连通区域的二值图像
        '''
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(bin_img)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_bin_img = np.zeros_like(bin_img)
        new_bin_img[labels == largest_label] = 255
        return new_bin_img

    def close_operation(self, bin_img, kernel_size=(5, 5)):
        '''
        对二值图像进行闭运算，用于消除内部小孔和连接接近的对象。
        :param bin_img: 输入的二值图像
        :param kernel_size: 核的大小
        :return: 进行闭运算后的图像
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        return closed_img

    def open_operation(self, bin_img, kernel_size=(5, 5)):
        '''
        对二值图像进行开运算，用于去除小的噪点。
        :param bin_img: 输入的二值图像
        :param kernel_size: 核的大小
        :return: 进行开运算后的图像
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opened_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        return opened_img

    def draw_tomato_edge(self, original_img, bin_img):
        '''
        在原始图像上绘制最大西红柿轮廓的近似多边形。
        :param original_img: 原始 BGR 图像
        :param bin_img: 西红柿的二值图像
        :return: 带有绘制边缘的原始图像和边缘掩码
        '''
        bin_img_processed = self.close_operation(bin_img, kernel_size=(15, 15))
        contours, _ = cv2.findContours(bin_img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return original_img, np.zeros_like(bin_img)
        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.0006 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(original_img, [approx], -1, (0, 255, 0), 3)
        mask = np.zeros_like(bin_img)
        cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
        return original_img, mask

    def draw_tomato_edge_convex_hull(self, original_img, bin_img):
        '''
        在原始图像上绘制最大西红柿轮廓的凸包。
        :param original_img: 原始 BGR 图像
        :param bin_img: 西红柿的二值图像
        :return: 带有绘制凸包的原始图像
        '''
        bin_img_blurred = cv2.GaussianBlur(bin_img, (5, 5), 0)
        contours, _ = cv2.findContours(bin_img_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return original_img
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        cv2.drawContours(original_img, [hull], -1, (0, 255, 0), 3)
        return original_img

    def fill_holes(self, bin_img):
        '''
        使用 floodFill 算法填充图像中的孔洞。
        :param bin_img: 输入的二值图像
        :return: 填充后的图像
        '''
        img_filled = bin_img.copy()
        height, width = bin_img.shape
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(img_filled, mask, (0, 0), 255)
        img_filled_inv = cv2.bitwise_not(img_filled)
        img_filled = cv2.bitwise_or(bin_img, img_filled_inv)
        return img_filled

    def bitwise_and_rgb_with_binary(self, rgb_img, bin_img):
        '''
        将 RGB 图像与二值图像进行按位与操作，用于将二值区域应用于原始图像。
        :param rgb_img: 原始 RGB 图像
        :param bin_img: 二值图像
        :return: 按位与后的结果图像
        '''
        bin_img_3channel = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(rgb_img, bin_img_3channel)
        return result

    def extract_max_connected_area(self, image, lower_hsv, upper_hsv):
        '''
        提取图像中满足 HSV 范围条件的最大连通区域，并填充孔洞。
        :param image: 输入的 BGR 图像
        :param lower_hsv: HSV 范围的下限
        :param upper_hsv: HSV 范围的上限
        :return: 处理后的图像
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_bin_img = np.zeros_like(mask)
        new_bin_img[labels == largest_label] = 255
        img_filled = new_bin_img.copy()
        height, width = new_bin_img.shape
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(img_filled, mask, (0, 0), 255)
        img_filled_inv = cv2.bitwise_not(img_filled)
        img_filled = cv2.bitwise_or(new_bin_img, img_filled_inv)
        return img_filled
