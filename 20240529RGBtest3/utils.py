# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 18:24
# @Author  : TG
# @File    : utils.py
# @Software: PyCharm

import time
import logging
import numpy as np
import shutil
import cv2
import os
from scipy.ndimage.measurements import label, find_objects
import win32pipe
import win32file
import io
from PIL import Image
import select
import msvcrt
from classifer import Tomato, Passion_fruit


def receive_rgb_data(pipe):
    try:
        # 读取图片数据长度
        len_img = win32file.ReadFile(pipe, 4, None)
        data_size = int.from_bytes(len_img[1], byteorder='big')
        # 读取实际图片数据
        result, data = win32file.ReadFile(pipe, data_size, None)
        # 检查读取操作是否成功
        if result != 0:
            print(f"读取失败，错误代码: {result}")
            return None
        # 返回成功读取的数据
        return data
    except Exception as e:
        print(f"数据接收失败，错误原因: {e}")
        return None

def receive_spec_data(pipe):
    try:
        # 读取光谱数据长度
        len_spec = win32file.ReadFile(pipe, 4, None)
        data_size = int.from_bytes(len_spec[1], byteorder='big')
        # 读取光谱数据
        result, spec_data = win32file.ReadFile(pipe, data_size, None)
        # 检查读取操作是否成功
        if result != 0:
            print(f"读取失败，错误代码: {result}")
            return None
        # 返回成功读取的数据
        return spec_data
    except Exception as e:
        print(f"数据接收失败，错误原因: {e}")
        return None

def parse_protocol(data: bytes) -> (str, any):
    """
    指令转换.

    :param data:接收到的报文
    :return: 指令类型和内容
    """
    try:
        assert len(data) > 2
    except AssertionError:
        logging.error('指令转换失败，长度不足3')
        return '', None
    cmd, data = data[:2], data[2:]
    cmd = cmd.decode('ascii').strip().upper()
    if cmd == 'TO':
        n_rows, n_cols, img = data[:2], data[2:4], data[4:]
        try:
            n_rows, n_cols = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols: {n_cols}')
            return '', None
        try:
            assert n_rows * n_cols * 3 == len(img)
            # 因为是float32类型 所以长度要乘12 ，如果是uint8则乘3
        except AssertionError:
            logging.error('图像指令IM转换失败，数据长度错误')
            return '', None
        img = np.frombuffer(img, dtype=np.uint8).reshape((n_rows, n_cols, -1))
        return cmd, img
    elif cmd == 'PF':
        n_rows, n_cols, n_bands, spec = data[:2], data[2:4], data[4:6], data[6:]
        try:
            n_rows, n_cols, n_bands = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols, n_bands]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols: {n_cols}, n_bands: {n_bands}')
            return '', None
        try:
            assert n_rows * n_cols * n_bands * 4 == len(spec)

        except AssertionError:
            logging.error('图像指令转换失败，数据长度错误')
            return '', None
        spec = np.frombuffer(spec, dtype=np.uint16).reshape(n_cols, n_rows, -1)
        return cmd, spec

def create_pipes(rgb_receive_name, rgb_send_name, spec_receive_name):
    while True:
        try:
            # 打开或创建命名管道
            rgb_receive = win32pipe.CreateNamedPipe(
                rgb_receive_name,
                win32pipe.PIPE_ACCESS_INBOUND,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                1, 80000000, 80000000, 0, None
            )
            rgb_send = win32pipe.CreateNamedPipe(
                rgb_send_name,
                win32pipe.PIPE_ACCESS_OUTBOUND,  # 修改为输出模式
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                1, 80000000, 80000000, 0, None
            )
            spec_receive = win32pipe.CreateNamedPipe(
                spec_receive_name,
                win32pipe.PIPE_ACCESS_INBOUND,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                1, 200000000, 200000000, 0, None
            )
            print("pipe管道创建成功，等待连接...")
            # 等待发送端连接
            win32pipe.ConnectNamedPipe(rgb_receive, None)
            print("rgb_receive connected.")
            # 等待发送端连接
            win32pipe.ConnectNamedPipe(rgb_send, None)
            print("rgb_send connected.")
            win32pipe.ConnectNamedPipe(rgb_receive, None)
            print("spec_receive connected.")
            return rgb_receive, rgb_send, spec_receive

        except Exception as e:
            print(f"管道创建连接失败，失败原因: {e}")
            print("等待5秒后重试...")
            time.sleep(5)
            continue

def send_data(pipe_send, long_axis, short_axis, defect_num, total_defect_area, rp):
    # start_time = time.time()
    #
    rp1 = Image.fromarray(rp.astype(np.uint8))
    # cv2.imwrite('rp1.bmp', rp1)

    # 将 Image 对象保存到 BytesIO 流中
    img_bytes = io.BytesIO()
    rp1.save(img_bytes, format='BMP')
    img_bytes = img_bytes.getvalue()

    # width = rp.shape[0]
    # height = rp.shape[1]
    # print(width, height)
    # img_bytes = rp.tobytes()
    # length = len(img_bytes) + 18
    # print(length)
    # length = length.to_bytes(4, byteorder='big')
    # width = width.to_bytes(2, byteorder='big')
    # height = height.to_bytes(2, byteorder='big')

    print(f'原始长度:', len(rp.tobytes()))
    print(f'发送长度:', len(img_bytes))

    long_axis = long_axis.to_bytes(2, byteorder='big')
    short_axis = short_axis.to_bytes(2, byteorder='big')
    defect_num = defect_num.to_bytes(2, byteorder='big')
    total_defect_area = int(total_defect_area).to_bytes(4, byteorder='big')
    length = (len(img_bytes) + 4).to_bytes(4, byteorder='big')
    # cmd_type = 'RIM'
    # result = result.encode('ascii')
    # send_message = b'\xaa' + length + (' ' + cmd_type).upper().encode('ascii') + long_axis + short_axis + defect_num + total_defect_area + width + height + img_bytes + b'\xff\xff\xbb'
    # send_message = long_axis + short_axis + defect_num + total_defect_area + img_bytes
    send_message = long_axis + short_axis + defect_num + total_defect_area + length + img_bytes
    # print(long_axis)
    # print(short_axis)
    # print(defect_num)
    # print(total_defect_area)
    # print(width)
    # print(height)

    try:
        win32file.WriteFile(pipe_send, send_message)
        time.sleep(0.01)
        print('发送成功')
        # print(len(send_message))
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False

    # end_time = time.time()
    # print(f'发送时间：{end_time - start_time}秒')

    return True



def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

def create_file(file_name):
    """
    创建文件
    :param file_name: 文件名
    :return: None
    """
    if os.path.exists(file_name):
        print("文件存在：%s" % file_name)
        return False
        # os.remove(file_name)  # 删除已有文件
    if not os.path.exists(file_name):
        print("文件不存在，创建文件：%s" % file_name)
        open(file_name, 'a').close()
        return True


class Logger(object):
    def __init__(self, is_to_file=False, path=None):
        self.is_to_file = is_to_file
        if path is None:
            path = "tomato.log"
        self.path = path
        create_file(path)

    def log(self, content):
        if self.is_to_file:
            with open(self.path, "a") as f:
                print(time.strftime("[%Y-%m-%d_%H-%M-%S]:"), file=f)
                print(content, file=f)
        else:
            print(content)


def contour_process(image_array):
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

def analyze_ellipse(image_array):
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

# 示例用法
# image_array = cv2.imread('path_to_your_image.bmp', cv2.IMREAD_GRAYSCALE)
# major_axis, minor_axis = analyze_ellipse(image_array)
# print(f"Major Axis: {major_axis}, Minor Axis: {minor_axis}")

# 加载新上传的图像进行分析
new_ellipse_image_path = '/mnt/data/未标题-2.png'
new_ellipse_image = cv2.imread(new_ellipse_image_path, cv2.IMREAD_GRAYSCALE)

# 使用上述函数进行分析
analyze_ellipse(new_ellipse_image)


def analyze_defect(image_array):
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


# 示例用法
# image_array = cv2.imread('path_to_your_image.bmp', cv2.IMREAD_GRAYSCALE)
# black_areas_count, total_pixels = analyze_black_areas_in_white(image_array)
# print(f"Number of black areas: {black_areas_count}, Total pixels in black areas: {


def analyze_tomato(img):
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
    s_l = tomato.extract_s_l(img)
    thresholded_s_l = tomato.threshold_segmentation(s_l, threshold_s_l)
    new_bin_img = tomato.largest_connected_component(thresholded_s_l)
    # 绘制西红柿边缘并获取缺陷信息
    edge, mask = tomato.draw_tomato_edge(img, new_bin_img)
    org_defect = tomato.bitwise_and_rgb_with_binary(edge, new_bin_img)
    # 获取西红柿的尺寸信息
    long_axis, short_axis = analyze_ellipse(mask)
    # 获取缺陷信息
    number_defects, total_pixels = analyze_defect(new_bin_img)
    # 将处理后的图像转换为 RGB 格式
    rp = cv2.cvtColor(org_defect, cv2.COLOR_BGR2RGB)
    return long_axis, short_axis, number_defects, total_pixels, rp


def analyze_passion_fruit(img, hue_value=37, hue_delta=10, value_target=25, value_delta=10):
    if img is None:
        print("Error: 无图像数据.")
        return None

    # 创建PassionFruit类的实例
    pf = Passion_fruit(hue_value=hue_value, hue_delta=hue_delta, value_target=value_target, value_delta=value_delta)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined_mask = pf.create_mask(hsv_image)
    combined_mask = pf.apply_morphology(combined_mask)
    max_mask = pf.find_largest_component(combined_mask)

    # if max_mask is None:
    #     print("No significant components found.")
    #     return None
    contour_mask = contour_process(max_mask)
    long_axis, short_axis = analyze_ellipse(contour_mask)
    number_defects, total_pixels = analyze_defect(max_mask)
    edge = pf.draw_contours_on_image(img, contour_mask)
    org_defect = pf.bitwise_and_rgb_with_binary(edge, max_mask)
    rp = cv2.cvtColor(org_defect, cv2.COLOR_BGR2RGB)

    return long_axis, short_axis, number_defects, total_pixels, rp
