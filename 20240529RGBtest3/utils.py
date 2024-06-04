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
from classifer import Tomato

def receive_rgb_data(pipe):
    try:
        # 读取图片数据
        len_img = win32file.ReadFile(pipe, 4, None)
        data_size = int.from_bytes(len_img[1], byteorder='big')
        result, img_data = win32file.ReadFile(pipe, data_size, None)
        return img_data
    except Exception as e:
        print(f"数据接收失败，错误原因: {e}")
        return None


def receive_spec_data(pipe):
    try:
        # 读取图片数据长度
        len_spec = win32file.ReadFile(pipe, 4, None)
        if len_spec is None:
            # 未能读取到数据长度,返回"0"
            return "0"
        data_size = int.from_bytes(len_spec[1], byteorder='big')
        if data_size == 0:
            # 接收到空数据,返回"0"
            return "0"

        # 读取图片数据
        result, spec_data = win32file.ReadFile(pipe, data_size, None)
        return spec_data
    except Exception as e:
        print(f"数据接收失败，错误原因: {e}")
        return '0'

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




def get_tomato_dimensions(edge_img):
    """
    根据边缘二值化轮廓图,计算果子的长径、短径和长短径比值。
    使用最小外接矩形和最小外接圆两种方法。

    参数:
    edge_img (numpy.ndarray): 边缘二值化轮廓图,背景为黑色,番茄区域为白色。

    返回:
    tuple: (长径, 短径, 长短径比值)
    """
    if edge_img is None or edge_img.any() == 0:
        return (0, 0)
    # 最小外接矩形
    rect = cv2.minAreaRect(cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    major_axis, minor_axis = rect[1]
    # aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

    # # 最小外接圆
    # (x, y), radius = cv2.minEnclosingCircle(
    #     cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    # diameter = 2 * radius
    # aspect_ratio_circle = 1.0

    return (max(major_axis, minor_axis), min(major_axis, minor_axis))

def get_defect_info(defect_img):
    """
    根据区域缺陷二值化轮廓图,计算缺陷区域的个数和总面积。

    参数:
    defect_img (numpy.ndarray): 番茄区域缺陷二值化轮廓图,背景为黑色,番茄区域为白色,缺陷区域为黑色连通域。

    返回:
    tuple: (缺陷区域个数, 缺陷区域像素面积，缺陷像素总面积)
    """
    # 检查输入是否为空
    if defect_img is None or defect_img.any() == 0:
        return (0, 0)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(defect_img, connectivity=4)
    max_area = max(stats[i, cv2.CC_STAT_AREA] for i in range(1, nb_components))
    areas = []
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if area != max_area:
            areas.append(area)
    number_defects = len(areas)
    total_pixels = sum(areas)
    return number_defects, total_pixels

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
    long_axis, short_axis = get_tomato_dimensions(mask)
    # 获取缺陷信息
    number_defects, total_pixels = get_defect_info(new_bin_img)
    # 将处理后的图像转换为 RGB 格式
    rp = cv2.cvtColor(org_defect, cv2.COLOR_BGR2RGB)
    return long_axis, short_axis, number_defects, total_pixels, rp
