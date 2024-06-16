# -*- coding: utf-8 -*-
# @Time    : 2024/6/4 21:34
# @Author  : GG
# @File    : classifer.py
# @Software: PyCharm


import cv2
import numpy as np
import logging
import os
import utils
from root_dir import ROOT_DIR
from sklearn.ensemble import RandomForestRegressor
import joblib

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

class Passion_fruit:
    def __init__(self, hue_value=37, hue_delta=10, value_target=25, value_delta=10):
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
        bin_img_3channel = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(rgb_img, bin_img_3channel)
        return result

class Spec_predict(object):
    def __init__(self, load_from=None, debug_mode=False, class_weight=None):
        if load_from is None:
            self.model = RandomForestRegressor(n_estimators=100)
        else:
            self.load(load_from)
        self.log = utils.Logger(is_to_file=debug_mode)
        self.debug_mode = debug_mode

    def load(self, path=None):
        if path is None:
            path = os.path.join(ROOT_DIR, 'models')
            model_files = os.listdir(path)
            if len(model_files) == 0:
                self.log.log("No model found!")
                return 1
            self.log.log("./ Models Found:")
            _ = [self.log.log("├--" + str(model_file)) for model_file in model_files]
            file_times = [model_file[6:-2] for model_file in model_files]
            latest_model = model_files[int(np.argmax(file_times))]
            self.log.log("└--Using the latest model: " + str(latest_model))
            path = os.path.join(ROOT_DIR, "models", str(latest_model))
        if not os.path.isabs(path):
            logging.warning('给的是相对路径')
            return -1
        if not os.path.exists(path):
            logging.warning('文件不存在')
            return -1
        with open(path, 'rb') as f:
            model_dic = joblib.load(f)
        self.model = model_dic['model']
        return 0

    def predict(self, data_x):
        '''
        对数据进行预测
        :param data_x: 波段选择后的数据
        :return: 预测结果二值化后的数据，0为背景，1为黄芪,2为杂质2，3为杂质1，4为甘草片，5为红芪
        '''
        data_y = self.model.predict(data_x)

        return data_y

# def get_tomato_dimensions(edge_img):
#     """
#     根据边缘二值化轮廓图,计算果子的长径、短径和长短径比值。
#     使用最小外接矩形和最小外接圆两种方法。
#
#     参数:
#     edge_img (numpy.ndarray): 边缘二值化轮廓图,背景为黑色,番茄区域为白色。
#
#     返回:
#     tuple: (长径, 短径, 长短径比值)
#     """
#     if edge_img is None or edge_img.any() == 0:
#         return (0, 0)
#     # 最小外接矩形
#     rect = cv2.minAreaRect(cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
#     major_axis, minor_axis = rect[1]
#     # aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
#
#     # # 最小外接圆
#     # (x, y), radius = cv2.minEnclosingCircle(
#     #     cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
#     # diameter = 2 * radius
#     # aspect_ratio_circle = 1.0
#
#     return (max(major_axis, minor_axis), min(major_axis, minor_axis))

# def get_defect_info(defect_img):
#     """
#     根据区域缺陷二值化轮廓图,计算缺陷区域的个数和总面积。
#
#     参数:
#     defect_img (numpy.ndarray): 番茄区域缺陷二值化轮廓图,背景为黑色,番茄区域为白色,缺陷区域为黑色连通域。
#
#     返回:
#     tuple: (缺陷区域个数, 缺陷区域像素面积，缺陷像素总面积)
#     """
#     # 检查输入是否为空
#     if defect_img is None or defect_img.any() == 0:
#         return (0, 0)
#
#     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(defect_img, connectivity=4)
#     max_area = max(stats[i, cv2.CC_STAT_AREA] for i in range(1, nb_components))
#     areas = []
#     for i in range(1, nb_components):
#         area = stats[i, cv2.CC_STAT_AREA]
#         if area != max_area:
#             areas.append(area)
#     number_defects = len(areas)
#     total_pixels = sum(areas)
#     return number_defects, total_pixels

class Data_processing:
    def __init__(self):
        pass

    def contour_process(self, image_array):
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

    def analyze_defect(self, image_array):
        # 查找白色区域的轮廓
        _, binary_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
        contours_white, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化统计数据
        count_black_areas = 0
        total_pixels_black_areas = 0

        # 对于每个白色区域，查找内部的黑色小区域
        for contour in contours_white:
            # 创建一个mask以查找内部的黑色区域
            mask = np.zeros_like(image_array)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # 仅在白色轮廓内部查找黑色区域
            black_areas_inside = cv2.bitwise_and(cv2.bitwise_not(image_array), mask)

            # 查找黑色区域的轮廓
            contours_black, _ = cv2.findContours(black_areas_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count_black_areas += len(contours_black)

            # 计算黑色区域的总像素数
            for c in contours_black:
                total_pixels_black_areas += cv2.contourArea(c)

        number_defects = count_black_areas
        total_pixels = total_pixels_black_areas
        return number_defects, total_pixels

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
        density = 0.652228972
        a = long_axis / 2
        b = short_axis /2
        volume = 4 / 3 * np.pi * a * b * b
        weigth = volume * density
        return weigth
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
        # 设置 S-L 通道阈值并处理图像
        threshold_s_l = 180
        threshold_fore_g_r_t = 20
        s_l = tomato.extract_s_l(img)
        thresholded_s_l = tomato.threshold_segmentation(s_l, threshold_s_l)
        new_bin_img = tomato.largest_connected_component(thresholded_s_l)
        # 绘制西红柿边缘并获取缺陷信息
        edge, mask = tomato.draw_tomato_edge(img, new_bin_img)
        org_defect = tomato.bitwise_and_rgb_with_binary(edge, new_bin_img)
        fore = tomato.bitwise_and_rgb_with_binary(img, mask)
        fore_g_r_t = tomato.threshold_segmentation(tomato.extract_g_r(fore), threshold=threshold_fore_g_r_t)
        # 统计白色像素点个数
        # print(np.sum(fore_g_r_t == 255))
        # print(np.sum(mask == 255))
        # print(np.sum(fore_g_r_t == 255) / np.sum(mask == 255))
        green_percentage = np.sum(fore_g_r_t == 255) / np.sum(mask == 255)
        green_percentage = round(green_percentage, 2) * 100
        # 获取西红柿的尺寸信息
        long_axis, short_axis = self.analyze_ellipse(mask)
        # 获取缺陷信息
        number_defects, total_pixels = self.analyze_defect(new_bin_img)
        # 将处理后的图像转换为 RGB 格式
        rp = cv2.cvtColor(org_defect, cv2.COLOR_BGR2RGB)
        diameter =  (long_axis + short_axis) / 2
        return diameter, green_percentage, number_defects, total_pixels, rp

    def analyze_passion_fruit(self, img, hue_value=37, hue_delta=10, value_target=25, value_delta=10):
        if img is None:
            print("Error: 无图像数据.")
            return None

        # 创建PassionFruit类的实例
        pf = Passion_fruit(hue_value=hue_value, hue_delta=hue_delta, value_target=value_target, value_delta=value_delta)

        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        combined_mask = pf.create_mask(hsv_image)
        combined_mask = pf.apply_morphology(combined_mask)
        max_mask = pf.find_largest_component(combined_mask)

        contour_mask = self.contour_process(max_mask)
        long_axis, short_axis = self.analyze_ellipse(contour_mask)
        weigth = self.weight_estimates(long_axis, short_axis)
        number_defects, total_pixels = self.analyze_defect(max_mask)
        edge = pf.draw_contours_on_image(img, contour_mask)
        org_defect = pf.bitwise_and_rgb_with_binary(edge, max_mask)
        rp = cv2.cvtColor(org_defect, cv2.COLOR_BGR2RGB)
        diameter = (long_axis + short_axis) / 2

        return diameter, weigth, number_defects, total_pixels, rp
