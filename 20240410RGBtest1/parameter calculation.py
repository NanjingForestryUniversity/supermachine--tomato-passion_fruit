# -*- coding: utf-8 -*-
# @Time    : 2024/4/11 15:18
# @Author  : TG
# @File    : parameter calculation.py
# @Software: PyCharm

import cv2
import numpy as np
from scipy.ndimage.measurements import label, find_objects

def get_tomato_dimensions(edge_img):
    """
    根据番茄边缘二值化轮廓图,计算番茄的长径、短径和长短径比值。
    使用最小外接矩形和最小外接圆两种方法。

    参数:
    edge_img (numpy.ndarray): 番茄边缘二值化轮廓图,背景为黑色,番茄区域为白色。

    返回:
    tuple: (长径, 短径, 长短径比值)
    """
    # 最小外接矩形
    rect = cv2.minAreaRect(cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    major_axis, minor_axis = rect[1]
    aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

    # # 最小外接圆
    # (x, y), radius = cv2.minEnclosingCircle(
    #     cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    # diameter = 2 * radius
    # aspect_ratio_circle = 1.0

    return (max(major_axis, minor_axis), min(major_axis, minor_axis), aspect_ratio)

def get_defect_info(defect_img):
    """
    根据番茄区域缺陷二值化轮廓图,计算缺陷区域的个数和总面积。

    参数:
    defect_img (numpy.ndarray): 番茄区域缺陷二值化轮廓图,背景为黑色,番茄区域为白色,缺陷区域为黑色连通域。

    返回:
    tuple: (缺陷区域个数, 缺陷区域像素面积，缺陷像素总面积)
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(defect_img, connectivity=4)
    max_area = max(stats[i, cv2.CC_STAT_AREA] for i in range(1, nb_components))
    areas = []
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if area != max_area:
            areas.append(area)
    number_defects = len(areas)
    total_pixels = sum(areas)
    return number_defects, areas, total_pixels




def connected_components_analysis(binary_image):
    """
    从二值化图像计算黑色连通域个数和各个黑色连通域像素面积及黑色像素总面积。

    参数:
    binary_image (numpy.ndarray): 二值化图像, 其中 0 表示白色, 1 表示黑色。

    返回:
    num_components (int): 黑色连通域的个数。
    component_areas (list): 每个黑色连通域的像素面积。
    total_black_area (int): 黑色像素的总面积。
    """
    # 标记连通域
    labeled_image, num_components = label(binary_image)

    # 获取每个连通域的像素位置
    slices = find_objects(labeled_image)

    # 计算每个连通域的像素面积
    component_areas = []
    for slice_obj in slices:
        component_area = np.sum(binary_image[slice_obj])
        component_areas.append(component_area)

    # 计算黑色像素的总面积
    total_black_area = np.sum(binary_image)

    return num_components, component_areas, total_black_area



def main():
    # 读取图像
    defect_image = cv2.imread(r'D:\project\Tomato\20240410tomatoRGBtest2\Largest Connected Component_screenshot_15.04.2024.png', 0)
    edge_image = cv2.imread(r'D:\project\Tomato\20240410tomatoRGBtest2\mask_screenshot_15.04.2024.png', 0)
    filled_image = cv2.imread(r'D:\project\Tomato\20240410tomatoRGBtest2\Filled_screenshot_15.04.2024.png', 0)

    # print(defect_image.shape)
    # print(edge_image.shape)
    # print(filled_image.shape)

    # 执行二值化处理
    _, thresh_defect = cv2.threshold(defect_image, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_edge = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY)
    _, thresh_filled = cv2.threshold(filled_image, 127, 255, cv2.THRESH_BINARY)

    print(thresh_defect.shape)
    print(thresh_edge.shape)
    print(thresh_filled.shape)

    # # 直接使用二值图像
    # thresh_defect = defect_image
    # thresh_edge = edge_image
    # thresh_filled = filled_image

    # 获取番茄的长径、短径和长短径比值
    major_axis, minor_axis, aspect_ratio = get_tomato_dimensions(thresh_edge)

    # 获取缺陷区域的个数和总面积
    num_defects, areas, total_pixels = get_defect_info(thresh_defect)

    # 获取黑色连通域的个数、各个连通域的面积和总黑色面积
    num_components, component_areas, total_black_area = connected_components_analysis(thresh_filled)

    print(f'番茄的长径为{major_axis},短径为{minor_axis},长短径比值为{aspect_ratio}')
    print(f'缺陷区域的个数为{num_defects},像素个数分别为{areas},缺陷总面积为{total_pixels}')
    print(f'黑色连通域的个数为{num_components},像素个数分别为{component_areas},黑色像素总面积为{total_black_area}')

if __name__ == '__main__':
    main()

