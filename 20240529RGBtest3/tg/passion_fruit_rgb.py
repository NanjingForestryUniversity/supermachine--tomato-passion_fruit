# -*- coding: utf-8 -*-
# @Time    : 2024/6/26 下午5:31
# @Author  : TG
# @File    : passion_fruit_rgb.py
# @Software: PyCharm
import os
import cv2
import numpy as np
import argparse
import logging
from config import Config as setting

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



def fill_holes(bin_img):

    img_filled = bin_img.copy()
    height, width = bin_img.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(img_filled, mask, (0, 0), 255)
    img_filled_inv = cv2.bitwise_not(img_filled)
    img_filled = cv2.bitwise_or(bin_img, img_filled)
    img_defect = img_filled_inv[:height, :width]
    return img_filled, img_defect

def contour_process(image_array):
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


def extract_green_pixels_cv(image):
    """
    使用 OpenCV 提取图像中的绿色像素，并可选择保存结果图像。

    参数:
        image_path (str): 输入图像的文件路径。
        save_path (str, optional): 输出图像的保存路径，若提供此参数，则保存提取的绿色像素图像。

    返回:
        输出图像，绿色像素为白色，其他像素为黑色。
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Define the HSV range for green
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([60, 180, 60])
     # Convert the image to HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Create the mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    # Convert result to BGR for display
    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return mask


def pixel_comparison(defect, mask):
    """
    比较两幅图像的像素值，如果相同则赋值为0，不同则赋值为255。
    参数:
        defect_path (str): 第一幅图像的路径。
        mask_path (str): 第二幅图像的路径。
        save_path (str, optional): 结果图像的保存路径。
    返回:
        numpy.ndarray: 处理后的图像数组。
    """
    # 确保图像是二值图像
    _, defect_binary = cv2.threshold(defect, 127, 255, cv2.THRESH_BINARY)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # 执行像素比较
    green_img = np.where(defect_binary == mask_binary, 0, 255).astype(np.uint8)
    return green_img


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir_path', type=str,
                        default=r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\tg\test',
                        help='the directory path of images')
    parser.add_argument('--threshold_s_l', type=int, default=180,
                        help='the threshold for s_l')
    parser.add_argument('--threshold_r_b', type=int, default=15,
                        help='the threshold for r_b')

    args = parser.parse_args()
    pf = Passion_fruit()

    for img_file in os.listdir(args.dir_path):
        if img_file.endswith('.bmp'):
            img_path = os.path.join(args.dir_path, img_file)
            img = cv2.imread(img_path)
            cv2.imshow('img', img)
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cv2.imshow('hsv', hsv_image)
            combined_mask = pf.create_mask(hsv_image)
            cv2.imshow('combined_mask1', combined_mask)
            combined_mask = pf.apply_morphology(combined_mask)
            cv2.imshow('combined_mask2', combined_mask)
            max_mask = pf.find_largest_component(combined_mask)
            cv2.imshow('max_mask', max_mask)

            filled_img, defect = fill_holes(max_mask)
            cv2.imshow('filled_img', filled_img)
            cv2.imshow('defect', defect)

            contour_mask = contour_process(max_mask)
            cv2.imshow('contour_mask', contour_mask)

            fore = pf.bitwise_and_rgb_with_binary(img, contour_mask)
            cv2.imshow('fore', fore)

            mask = extract_green_pixels_cv(fore)
            cv2.imshow('mask', mask)

            green_img = pixel_comparison(defect, mask)
            cv2.imshow('green_img', green_img)

            green_percentage = np.sum(green_img == 255) / np.sum(contour_mask == 255)
            green_percentage = round(green_percentage, 2)

            print(np.sum(green_img == 255))
            print(np.sum(contour_mask == 255))
            print(green_percentage)




            edge = pf.draw_contours_on_image(img, contour_mask)
            cv2.imshow('edge', edge)
            org_defect = pf.bitwise_and_rgb_with_binary(edge, max_mask)
            cv2.imshow('org_defect', org_defect)


            cv2.waitKey(0)
            cv2.destroyAllWindows()











if __name__ == '__main__':
    main()