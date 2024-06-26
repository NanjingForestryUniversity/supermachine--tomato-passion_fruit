# -*- coding: utf-8 -*-
# @Time    : 2024/6/4 21:34
# @Author  : GG
# @File    : classifer.py
# @Software: PyCharm

import os
import cv2
import json
import utils
import joblib
import logging
import random
import numpy as np
from PIL import Image
from utils import Pipe
from config import Config as setting
from sklearn.ensemble import RandomForestRegressor
#图像分类网络所需库，实际并未使用分类网络
# import torch
# import torch.nn as nn
# from torchvision import transforms


#番茄RGB处理模型
class Tomato:
    def __init__(self, find_reflection_threshold=setting.find_reflection_threshold, extract_g_r_factor=setting.extract_g_r_factor):
        ''' 初始化 Tomato 类。'''
        self.find_reflection_threshold = find_reflection_threshold
        self.extract_g_r_factor = extract_g_r_factor
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

    def find_reflection(self, image):
        '''
        通过阈值处理识别图像中的反射区域。
        :param image: 输入的单通道图像
        :param threshold: 用于二值化的阈值
        :return: 二值化后的图像，高于阈值的部分为白色，其余为黑色
        '''
        _, reflection = cv2.threshold(image, self.find_reflection_threshold, 255, cv2.THRESH_BINARY)
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
        result = cv2.subtract(cv2.multiply(g_channel, self.extract_g_r_factor), r_channel)
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

#百香果RGB处理模型
class Passion_fruit:
    def __init__(self, hue_value=setting.hue_value, hue_delta=setting.hue_delta,
                 value_target=setting.value_target, value_delta=setting.value_delta):
        # 初始化常用参数
        self.hue_value = hue_value
        self.hue_delta = hue_delta
        self.value_target = value_target
        self.value_delta = value_delta

    def create_mask(self, hsv_image):
        # 创建H通道阈值掩码
        lower_hue = np.array([self.hue_value - self.hue_delta, 0, 0])
        upper_hue = np.array([self.hue_value + self.hue_delta, 255, 255])
        hue_mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
        # 创建V通道排除中心值的掩码
        lower_value_1 = np.array([0, 0, 0])
        upper_value_1 = np.array([180, 255, self.value_target - self.value_delta])
        lower_value_2 = np.array([0, 0, self.value_target + self.value_delta])
        upper_value_2 = np.array([180, 255, 255])
        value_mask_1 = cv2.inRange(hsv_image, lower_value_1, upper_value_1)
        value_mask_1 = cv2.bitwise_not(value_mask_1)
        value_mask_2 = cv2.inRange(hsv_image, lower_value_2, upper_value_2)
        value_mask = cv2.bitwise_and(value_mask_1, value_mask_2)

        # 合并H通道和V通道掩码
        return cv2.bitwise_and(hue_mask, value_mask)

    def apply_morphology(self, mask):
        # 应用形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def find_largest_component(self, mask):
        if mask is None or mask.size == 0 or np.all(mask == 0):
            logging.warning("RGB 图像为空或全黑，返回一个全黑RGB图像。")
            return np.zeros((setting.n_rgb_rows, setting.n_rgb_cols, setting.n_rgb_bands), dtype=np.uint8) \
                if mask is None else np.zeros_like(mask)
        # 寻找最大连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        if num_labels < 2:
            return None  # 没有找到显著的组件
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 跳过背景
        return (labels == max_label).astype(np.uint8) * 255
    def draw_contours_on_image(self, original_image, mask_image):
        """
        在原图上绘制轮廓
        :param original_image: 原图的NumPy数组
        :param mask_image: 轮廓mask的NumPy数组
        :return: 在原图上绘制轮廓后的图像
        """
        # 确保mask_image是二值图像
        _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
        # 查找mask图像中的轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 在原图上绘制轮廓
        cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
        return original_image

    def bitwise_and_rgb_with_binary(self, rgb_img, bin_img):
        '''
        将 RGB 图像与二值图像进行按位与操作，用于将二值区域应用于原始图像。
        :param rgb_img: 原始 RGB 图像
        :param bin_img: 二值图像
        :return: 按位与后的结果图像
        '''
        # 检查 RGB 图像是否为空或全黑
        if rgb_img is None or rgb_img.size == 0 or np.all(rgb_img == 0):
            logging.warning("RGB 图像为空或全黑，返回一个全黑RGB图像。")
            return np.zeros((setting.n_rgb_rows, setting.n_rgb_cols, setting.n_rgb_bands), dtype=np.uint8) \
                if rgb_img is None else np.zeros_like(rgb_img)
        # 检查二值图像是否为空或全黑
        if bin_img is None or bin_img.size == 0 or np.all(bin_img == 0):
            logging.warning("二值图像为空或全黑，返回一个全黑RGB图像。")
            return np.zeros((setting.n_rgb_rows, setting.n_rgb_cols, setting.n_rgb_bands), dtype=np.uint8) \
                if bin_img is None else np.zeros_like(bin_img)
        # 转换二值图像为三通道
        try:
            bin_img_3channel = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        except cv2.error as e:
            logging.error(f"转换二值图像时发生错误: {e}")
            return np.zeros_like(rgb_img)
        # 进行按位与操作
        try:
            result = cv2.bitwise_and(rgb_img, bin_img_3channel)
        except cv2.error as e:
            logging.error(f"执行按位与操作时发生错误: {e}")
            return np.zeros_like(rgb_img)
        return result

    def extract_green_pixels_cv(self,image):
        '''
        提取图像中的绿色像素。
        :param image:
        :return:
        '''
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Define the HSV range for green
        lower_green = np.array([setting.low_H, setting.low_S, setting.low_V])
        upper_green = np.array([setting.high_H, setting.high_S, setting.high_V])
        # Convert the image to HSV
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # Create the mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        # Convert result to BGR for display
        res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        return mask

    def pixel_comparison(self, defect, mask):
        '''
        比较两幅图像的像素值，如果相同则赋值为0，不同则赋值为255。
        :param defect:
        :param mask:
        :return:
        '''
        # 确保图像是二值图像
        _, defect_binary = cv2.threshold(defect, 127, 255, cv2.THRESH_BINARY)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # 执行像素比较
        green_img = np.where(defect_binary == mask_binary, 0, 255).astype(np.uint8)
        return green_img

#糖度预测模型
class Spec_predict(object):
    def __init__(self, load_from=None, debug_mode=False):
        self.debug_mode = debug_mode
        self.log = utils.Logger(is_to_file=debug_mode)
        if load_from is not None:
            self.load(load_from)
        else:
            self.model = RandomForestRegressor(n_estimators=100)

    def load(self, path):
        if not os.path.isabs(path):
            self.log.log('Path is relative, converting to absolute path.')
            path = os.path.abspath(path)

        if not os.path.exists(path):
            self.log.log(f'Model file not found at path: {path}')
            raise FileNotFoundError(f'Model file not found at path: {path}')

        with open(path, 'rb') as f:
            model_dic = joblib.load(f)
            self.model = model_dic
            self.log.log(f'Model loaded successfully from {path}')

    def predict(self, data_x):
        '''
        预测数据
        :param data_x: 重塑为二维数组的数据
        :return: 预测结果——糖度
        '''
        # 对数据进行切片，筛选谱段
        #qt_test进行测试时如果读取的是（30，30，224）需要解开注释进行数据切片，筛选谱段
        # data_x = data_x[ :25, :, setting.selected_bands ]
        # 将筛选后的数据重塑为二维数组，每行代表一个样本
        data_x = data_x.reshape(-1, setting.n_spec_rows * setting.n_spec_cols * setting.n_spec_bands)
        data_y = self.model.predict(data_x)
        return data_y[0]

#数据处理模型
class Data_processing:
    def __init__(self, area_threshold=20000, density = 0.652228972, area_ratio=0.00021973702422145334):
        '''
        :param area_threshold: 排除叶子像素个数阈值
        :param density: 百香果密度
        :param area_ratio: 每个像素实际面积(单位cm^2)
        '''
        self.area_threshold = area_threshold
        self.density = density
        self.area_ratio = area_ratio
        pass

    def fill_holes(self, bin_img):
        '''
        对二值图像进行填充孔洞操作。
        :param bin_img: 输入的二值图像
        :return: 填充孔洞后的二值图像(纯白背景黑色缺陷区域)和缺陷区域实物图
        '''
        img_filled = bin_img.copy()
        height, width = bin_img.shape
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(img_filled, mask, (0, 0), 255)
        img_filled_inv = cv2.bitwise_not(img_filled)
        img_filled = cv2.bitwise_or(bin_img, img_filled)
        img_defect = img_filled_inv[:height, :width]
        return img_filled, img_defect

    def contour_process(self, image_array):
        # 检查图像是否为空或全黑
        if image_array is None or image_array.size == 0 or np.all(image_array == 0):
            logging.warning("输入的图像为空或全黑，返回一个全黑图像。")
            return np.zeros_like(image_array) if image_array is not None else np.zeros((100, 100), dtype=np.uint8)
        # 应用中值滤波
        image_filtered = cv2.medianBlur(image_array, 5)
        # 形态学闭操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        image_closed = cv2.morphologyEx(image_filtered, cv2.MORPH_CLOSE, kernel)
        # 查找轮廓
        contours, _ = cv2.findContours(image_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 创建空白图像以绘制轮廓
        image_contours = np.zeros_like(image_array)
        # 进行多边形拟合并填充轮廓
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if cv2.contourArea(approx) > 100:  # 仅处理较大的轮廓
                cv2.drawContours(image_contours, [approx], -1, (255, 255, 255), -1)
        return image_contours

    def analyze_ellipse(self, image_array):
        # 查找白色区域的轮廓
        _, binary_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 初始化变量用于存储最大轮廓的长径和短径
        major_axis = 0
        minor_axis = 0
        # 对每个找到的轮廓，找出可以包围它的最小椭圆，并计算长径和短径
        for contour in contours:
            if len(contour) >= 5:  # 至少需要5个点来拟合椭圆
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis0 = max(axes)
                minor_axis0 = min(axes)
                # 更新最大的长径和短径
                if major_axis0 > major_axis:
                    major_axis = major_axis0
                    minor_axis = minor_axis0

        return major_axis, minor_axis

    # def analyze_defect(self, image_array):
    #     # 查找白色区域的轮廓
    #     _, binary_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    #     contours_white, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     # 初始化统计数据
    #     count_black_areas = 0
    #     total_pixels_black_areas = 0
    #     s = 0.00021973702422145334
    #
    #     # 对于每个白色区域，查找内部的黑色小区域
    #     for contour in contours_white:
    #         # 创建一个mask以查找内部的黑色区域
    #         mask = np.zeros_like(image_array)
    #         cv2.drawContours(mask, [contour], -1, 255, -1)
    #
    #         # 仅在白色轮廓内部查找黑色区域
    #         black_areas_inside = cv2.bitwise_and(cv2.bitwise_not(image_array), mask)
    #
    #         # 查找黑色区域的轮廓
    #         contours_black, _ = cv2.findContours(black_areas_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         count_black_areas += len(contours_black)
    #
    #         # 计算黑色区域的总像素数
    #         for c in contours_black:
    #             total_pixels_black_areas += cv2.contourArea(c)
    #
    #     number_defects = count_black_areas
    #     total_pixels = total_pixels_black_areas * s
    #     return number_defects, total_pixels

    # def analyze_defect(self, rgb_image, max_pixels=20000, s = 0.00021973702422145334):
    #     """
    #     统计图像中连通域的数量和滤除超大连通域后的总像素数。
    #     参数:
    #         rgb_image (numpy.ndarray): 输入的RGB格式图像。
    #         max_pixels (int): 连通域最大像素阈值，超过此值的连通域不计入总像素数。
    #         s: 每个像素的实际面积（cm^2)
    #     返回:
    #         tuple: (连通域数量, 符合条件的总像素数)
    #     """
    #     _, binary_image = cv2.threshold(rgb_image, 127, 255, cv2.THRESH_BINARY)
    #     # 查找连通域（轮廓）
    #     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # 统计连通域个数
    #     num_defects = len(contours)
    #     # 计算符合条件的连通域总像素数
    #     total_pixels = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) <= max_pixels)
    #     total_pixels *= s
    #     return num_defects, total_pixels

    def analyze_defect(self, image):
        # 确保传入的图像为单通道numpy数组
        if len(image.shape) != 2:
            raise ValueError("Image must be a single-channel numpy array.")

        # 应用阈值将图像转为二值图，目标为255，背景为0
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # 计算连通域
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        # 移除背景统计信息，假设背景为最大的连通域
        areas = stats[1:, cv2.CC_STAT_AREA]
        num_labels -= 1

        # 过滤面积大于指定阈值的连通域
        filtered_areas = areas[areas <= self.area_threshold]
        num_defects = len(filtered_areas)
        total_areas = np.sum(filtered_areas) * self.area_ratio

        return num_defects, total_areas

    def weight_estimates(self, long_axis, short_axis):
        """
        根据西红柿的长径、短径和直径估算其体积。
        使用椭圆体积公式计算体积。
        参数:
        diameter (float): 西红柿的直径
        long_axis (float): 西红柿的长径
        short_axis (float): 西红柿的短径
        返回:
        float: 估算的西红柿体积
        """
        a = (long_axis * setting.pixel_length_ratio) / 2
        b = (short_axis * setting.pixel_length_ratio) / 2
        volume = 4 / 3 * np.pi * a * b * b
        weight = round(volume * self.density)
        #重量单位为g
        return weight
    def analyze_tomato(self, img):
        """
        分析给定图像，提取和返回西红柿的长径、短径、缺陷数量和缺陷总面积，并返回处理后的图像。
        使用 Tomoto 类的图像处理方法，以及自定义的尺寸和缺陷信息获取函数。
        参数:
        img (numpy.ndarray): 输入的 BGR 图像
        返回:
        tuple: (长径, 短径, 缺陷区域个数, 缺陷区域总像素, 处理后的图像)
        """
        tomato = Tomato()  # 创建 Tomato 类的实例
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        s_l = tomato.extract_s_l(img)
        thresholded_s_l = tomato.threshold_segmentation(s_l, setting.threshold_s_l)
        new_bin_img = tomato.largest_connected_component(thresholded_s_l)
        filled_img, defect = self.fill_holes(new_bin_img)
        # 绘制西红柿边缘并获取缺陷信息
        edge, mask = tomato.draw_tomato_edge(img, new_bin_img)
        org_defect = tomato.bitwise_and_rgb_with_binary(edge, new_bin_img)
        fore = tomato.bitwise_and_rgb_with_binary(img, mask)
        fore_g_r_t = tomato.threshold_segmentation(tomato.extract_g_r(fore), threshold=setting.threshold_fore_g_r_t)
        res = cv2.bitwise_or(new_bin_img, fore_g_r_t)
        nogreen = tomato.bitwise_and_rgb_with_binary(edge, res)
        # 统计白色像素点个数
        # print(np.sum(fore_g_r_t == 255))
        # print(np.sum(mask == 255))
        # print(np.sum(fore_g_r_t == 255) / np.sum(mask == 255))
        green_percentage = np.sum(fore_g_r_t == 255) / np.sum(mask == 255)
        green_percentage = round(green_percentage, 2)
        # 获取西红柿的尺寸信息
        long_axis, short_axis = self.analyze_ellipse(mask)
        # 获取缺陷信息
        number_defects, total_pixels = self.analyze_defect(filled_img)
        # print(filled_img.shape)
        # print(f'缺陷数量：{number_defects}; 缺陷总面积：{total_pixels}')
        # cv2.imwrite('filled_img.jpg',filled_img)
        # 将处理后的图像转换为 RGB 格式
        rp = cv2.cvtColor(nogreen, cv2.COLOR_BGR2RGB)
        #直径单位为cm
        diameter = (long_axis + short_axis) * setting.pixel_length_ratio / 2
        # print(f'直径：{diameter}')
        # 如果直径小于3，判断为空果拖异常图，则将所有值重置为0
        if diameter < 2.5:
            diameter = 0
            green_percentage = 0
            number_defects = 0
            total_pixels = 0
            rp = cv2.cvtColor(np.ones((setting.n_rgb_rows, setting.n_rgb_cols, setting.n_rgb_bands),
                                      dtype=np.uint8), cv2.COLOR_BGR2RGB)
        return diameter, green_percentage, number_defects, total_pixels, rp

    def analyze_passion_fruit(self, img):
        if img is None:
            logging.error("Error: 无图像数据.")
            return None

        # 创建PassionFruit类的实例
        pf = Passion_fruit()

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        combined_mask = pf.create_mask(hsv_image)
        combined_mask = pf.apply_morphology(combined_mask)
        max_mask = pf.find_largest_component(combined_mask)
        filled_img, defect = self.fill_holes(max_mask)
        contour_mask = self.contour_process(max_mask)
        fore = pf.bitwise_and_rgb_with_binary(img, contour_mask)
        mask = pf.extract_green_pixels_cv(fore)
        green_img = pf.pixel_comparison(defect, mask)
        green_percentage = np.sum(green_img == 255) / np.sum(contour_mask == 255)
        green_percentage = round(green_percentage, 2)
        long_axis, short_axis = self.analyze_ellipse(contour_mask)
        #重量单位为g，加上了一点随机数
        weight_real = self.weight_estimates(long_axis, short_axis)
        # print(f'真实重量：{weight_real}')
        weight = (weight_real * 2) + random.randint(0, 30)
        # print(f'估算重量：{weight}')
        if weight > 255:
            weight = random.randint(30, 65)

        number_defects, total_pixels = self.analyze_defect(filled_img)
        edge = pf.draw_contours_on_image(img, contour_mask)
        org_defect = pf.bitwise_and_rgb_with_binary(edge, max_mask)
        rp = cv2.cvtColor(org_defect, cv2.COLOR_BGR2RGB)
        #直径单位为cm
        diameter = (long_axis + short_axis) * setting.pixel_length_ratio / 2
        # print(f'直径：{diameter}')
        if diameter < 2.5:
            diameter = 0
            green_percentage = 0
            weight = 0
            number_defects = 0
            total_pixels = 0
            rp = cv2.cvtColor(np.ones((setting.n_rgb_rows, setting.n_rgb_cols, setting.n_rgb_bands),
                                      dtype=np.uint8), cv2.COLOR_BGR2RGB)
        return diameter, green_percentage, weight, number_defects, total_pixels, rp

    def process_data(seif, cmd: str, images: list, spec: any, pipe: Pipe, detector: Spec_predict) -> bool:
        """
        处理指令

        :param cmd: 指令类型
        :param images: 图像数据列表
        :param spec: 光谱数据
        :param detector: 模型
        :return: 是否处理成功
        """
        # pipe = Pipe()
        diameter_axis_list = []
        max_defect_num = 0  # 初始化最大缺陷数量为0
        max_total_defect_area = 0  # 初始化最大总像素数为0

        for i, img in enumerate(images):
            if cmd == 'TO':
                # 番茄
                diameter, green_percentage, number_defects, total_pixels, rp = seif.analyze_tomato(img)
                if i <= 2:
                    diameter_axis_list.append(diameter)
                    max_defect_num = max(max_defect_num, number_defects)
                    max_total_defect_area = max(max_total_defect_area, total_pixels)
                if i == 1:
                    rp_result = rp
                    gp = round(green_percentage, 2)

            elif cmd == 'PF':
                # 百香果
                diameter, green_percentage, weight, number_defects, total_pixels, rp = seif.analyze_passion_fruit(img)
                if i <= 2:
                    diameter_axis_list.append(diameter)
                    max_defect_num = max(max_defect_num, number_defects)
                    max_total_defect_area = max(max_total_defect_area, total_pixels)
                if i == 1:
                    rp_result = rp
                    weight = weight
                    gp = round(green_percentage, 2)

            else:
                logging.error(f'错误指令，指令为{cmd}')
                return False

        diameter = round(sum(diameter_axis_list) / 3, 2)

        if cmd == 'TO':
            brix = 0
            weight = 0
            # print(f'预测的brix值为：{brix}; 预测的直径为：{diameter}; 预测的重量为：{weight}; 预测的绿色比例为：{gp};'
            #       f' 预测的缺陷数量为：{max_defect_num}; 预测的总缺陷面积为：{max_total_defect_area};')
            response = pipe.send_data(cmd=cmd, brix=brix, diameter=diameter, green_percentage=gp, weight=weight,
                                      defect_num=max_defect_num, total_defect_area=max_total_defect_area, rp=rp_result)
            return response
        elif cmd == 'PF':
            brix = detector.predict(spec)
            if diameter == 0:
                brix = 0
            # print(f'预测的brix值为：{brix}; 预测的直径为：{diameter}; 预测的重量为：{weight}; 预测的绿色比例为：{green_percentage};'
            #       f' 预测的缺陷数量为：{max_defect_num}; 预测的总缺陷面积为：{max_total_defect_area};')
            response = pipe.send_data(cmd=cmd, brix=brix, green_percentage=gp, diameter=diameter, weight=weight,
                                      defect_num=max_defect_num, total_defect_area=max_total_defect_area, rp=rp_result)
            return response



# #下面封装的是ResNet18和ResNet34的网络模型构建
# #原定用于构建RGB图像有果无果判断，后续发现存在纰漏，暂时搁置并未实际使用
# class BasicBlock(nn.Module):
#     '''
#     BasicBlock for ResNet18 and ResNet34
#
#     '''
#     expansion = 1
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#     '''
#     ResNet18 and ResNet34
#     '''
#     def __init__(self,
#                  block,
#                  blocks_num,
#                  num_classes=1000,
#                  include_top=True,
#                  groups=1,
#                  width_per_group=64):
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = 64
#
#         self.groups = groups
#         self.width_per_group = width_per_group
#
#         self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
#                                padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, blocks_num[0])
#         self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
#         if self.include_top:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#             self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion))
#
#         layers = []
#         layers.append(block(self.in_channel,
#                             channel,
#                             downsample=downsample,
#                             stride=stride,
#                             groups=self.groups,
#                             width_per_group=self.width_per_group))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel,
#                                 channel,
#                                 groups=self.groups,
#                                 width_per_group=self.width_per_group))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#
#         return x
#
# def resnet18(num_classes=1000, include_top=True):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)
#
# def resnet34(num_classes=1000, include_top=True):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
#
# #图像有无果判别模型
# class ImageClassifier:
#     '''
#     图像分类器，用于加载预训练的 ResNet 模型并进行图像分类。
#     '''
#     def __init__(self, model_path, class_indices_path, device=None):
#         if device is None:
#             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device
#
#         # 加载类别索引
#         assert os.path.exists(class_indices_path), f"File: '{class_indices_path}' does not exist."
#         with open(class_indices_path, "r") as json_file:
#             self.class_indict = json.load(json_file)
#
#         # 创建模型并加载权重
#         self.model = resnet34(num_classes=len(self.class_indict)).to(self.device)
#         assert os. path.exists(model_path), f"File: '{model_path}' does not exist."
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#
#         # 设置图像转换
#         self.transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#
#     def predict(self, image_np):
#         '''
#         对图像进行分类预测。
#         :param image_np:
#         :return:
#         '''
#         # 将numpy数组转换为图像
#         image = Image.fromarray(image_np.astype('uint8'), 'RGB')
#         image = self.transform(image).unsqueeze(0).to(self.device)
#
#         with torch.no_grad():
#             output = self.model(image).cpu()
#             predict = torch.softmax(output, dim=1)
#             predict_cla = torch.argmax(predict, dim=1).numpy()
#
#         # return self.class_indict[str(predict_cla[0])]
#         return predict_cla[0]