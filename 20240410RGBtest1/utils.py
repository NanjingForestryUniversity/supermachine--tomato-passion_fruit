# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 14:53
# @Author  : TG
# @File    : utils.py
# @Software: PyCharm

import os
import socket
import time
import logging
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
import pickle
import sys
import os
from scipy.ndimage.measurements import label, find_objects


class PreSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_pack = b''
        self.settimeout(5)

    def receive(self, *args, **kwargs):
        if self.pre_pack == b'':
            return self.recv(*args, **kwargs)
        else:
            data_len = args[0]
            required, left = self.pre_pack[:data_len], self.pre_pack[data_len:]
            self.pre_pack = left
            return required

    def set_prepack(self, pre_pack: bytes):
        temp = self.pre_pack
        self.pre_pack = temp + pre_pack


class DualSock(PreSocket):
    def __init__(self, connect_ip='127.0.0.1', recv_port: int = 21122, send_port: int = 21123):
        super().__init__()
        received_status, self.received_sock = try_connect(connect_ip=connect_ip, port_number=recv_port)
        send_status, self.send_sock = try_connect(connect_ip=connect_ip, port_number=send_port)
        self.status = received_status and send_status

    def send(self, *args, **kwargs) -> int:
        return self.send_sock.send(*args, **kwargs)

    def receive(self, *args, **kwargs) -> bytes:
        return self.received_sock.receive(*args, **kwargs)

    def set_prepack(self, pre_pack: bytes):
        self.received_sock.set_prepack(pre_pack)

    def reconnect(self, connect_ip='127.0.0.1', recv_port:int = 21122, send_port: int = 21123):
        received_status, self.received_sock = try_connect(connect_ip=connect_ip, port_number=recv_port)
        send_status, self.send_sock = try_connect(connect_ip=connect_ip, port_number=send_port)
        return received_status and send_status


def receive_sock(recv_sock: PreSocket, pre_pack: bytes = b'', time_out: float = -1.0, time_out_single=5e20) -> (
bytes, bytes):
    """
    从指定的socket中读取数据.自动阻塞，如果返回的数据为空则说明连接出现问题，需要重新连接。

    :param recv_sock: 指定sock
    :param pre_pack: 上一包的粘包内容
    :param time_out: 每隔time_out至少要发来一次指令,否则认为出现问题进行重连，小于0则为一直等
    :param time_out_single: 单次指令超时时间，单位是秒
    :return: data, next_pack
    """
    recv_sock.set_prepack(pre_pack)
    # 开头校验
    time_start_recv = time.time()
    while True:
        if time_out > 0:
            if (time.time() - time_start_recv) > time_out:
                logging.error(f'指令接收超时')
                return b'', b''
        try:
            temp = recv_sock.receive(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return b'', b''
        except TimeoutError as e:
            # logging.error(f'超时了，错误代码: \n{e}')
            logging.info('运行中,等待指令..')
            continue
        except socket.timeout as e:
            logging.info('运行中,等待指令..')
            continue
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return b'', b''
        if temp == b'\xaa':
            break

    # 接收开头后，开始进行时间记录
    time_start_recv = time.time()

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文的长度不正确, 错误代码: \n{e}')
            return b'', b''
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}')
        return b'', b''

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}')
            return b'', b''
    data, next_pack = temp[:data_len], temp[data_len:]
    recv_sock.set_prepack(next_pack)
    next_pack = b''

    # 进行数据校验
    temp = b''
    while len(temp) < 3:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文校验失败, 错误代码: \n{e}')
            return b'', b''
    if temp == b'\xff\xff\xbb':
        return data, next_pack
    else:
        logging.error(f"接收了一个完美的只错了校验位的报文")
        return b'', b''


def parse_protocol(data: bytes) -> (str, any):
    '''
    指令转换
    :param data: 接收到的报文
    :return: 指令类型，指令内容
    '''
    try:
        assert len(data) > 4
    except AssertionError:
        logging.error('指令转换失败，长度不足5')
        return '', None
    cmd, data = data[:4], data[4:]
    cmd = cmd.decode('ascii').strip().upper()
    if cmd == 'IM':
        n_rows, n_cols, img = data[:2], data[2:4], data[4:]
        try:
            n_rows, n_cols = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols: {n_cols}')
            return '', None
        try:
            assert n_rows * n_cols * 3 == len(img)
        except AssertionError:
            logging.error('图像指令IM转换失败，数据长度错误')
            return '', None
        img = np.frombuffer(img, dtype=np.uint8).reshape((n_rows, n_cols, -1))
        return cmd, img


def ack_sock(send_sock: socket.socket, cmd_type: str) -> bool:
    '''
    发送应答
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :return:是否发送成功
    '''
    msg = b'\xaa\x00\x00\x00\x05' + (' A' + cmd_type).upper().encode('ascii') + b'\xff\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送应答失败，错误类型：{e}')
        return False
    return True


def done_sock(send_sock: socket.socket, cmd_type: str, result = '') -> bool:
    '''
    发送任务完成指令
    :param send_sock: 指定sock
    :param cmd_type: 指令类型
    :param result: 数据
    :return: 是否发送成功
    '''
    cmd = cmd_type.strip().upper()
    if cmd_type == 'IM':
        result = result.encode()
        # 指令4位
        length = len(result) + 4
        length = length.to_bytes(4, byteorder='big')
        msg = b'\xaa' + length + (' D' + cmd).upper().encode('ascii') + result + b'\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True

def simple_sock(send_sock: socket.socket, cmd_type: str, result) -> bool:
    '''
    发送任务完成指令
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :param result:数据
    :return:是否发送成功
    '''
    cmd_type = cmd_type.strip().upper()
    if cmd_type == 'IM':
        if result == 0:
            msg = b'S'
        elif result == 1:
            msg = b'Z'
        elif result == 2:
            msg = b'Q'
    elif cmd_type == 'TR':
        msg = b'A'
    elif cmd_type == 'MD':
        msg = b'D'
    elif cmd_type == 'KM':
        msg = b'K'
        result = result.encode('ascii')
        result = b',' + result
        length = len(result)
        msg = msg + length.to_bytes(4, 'big') + result
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True

def test_sock(send_sock: socket.socket, cmd_type: str, long_axis, short_axis, defect_num, total_defect_area, rp) -> bool:
    start_time = time.time()

    cmd_type = cmd_type.strip().upper()
    if cmd_type == 'IM':
        # image_dir = r'D:\project\Tomato\20240410tomatoRGBtest2\huifu'
        # image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
        # for image_path in image_paths:
        #     img = cv2.imread(r'D:\project\Tomato\20240410tomatoRGBtest2\huifu\thresholded_6.jpg')
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     img = np.asarray(img, dtype=np.uint8)
            width = rp.shape[0]
            height = rp.shape[1]
            # print(width, height)
            img_bytes = rp.tobytes()
            length = len(img_bytes) + 18
            # print(length)
            length = length.to_bytes(4, byteorder='big')
            width = width.to_bytes(2, byteorder='big')
            height = height.to_bytes(2, byteorder='big')
            long_axis = long_axis.to_bytes(2, byteorder='big')
            short_axis = short_axis.to_bytes(2, byteorder='big')
            defect_num = defect_num.to_bytes(2, byteorder='big')
            total_defect_area = int(total_defect_area).to_bytes(4, byteorder='big')
            cmd_type = 'RIM'
            # result = result.encode('ascii')
            send_message = b'\xaa' + length + (' ' + cmd_type).upper().encode('ascii') + long_axis + short_axis + defect_num + total_defect_area + width + height + img_bytes + b'\xff\xff\xbb'
            # print(long_axis)
            # print(short_axis)
            # print(defect_num)
            # print(total_defect_area)
            # print(width)
            # print(height)

    try:
        send_sock.send(send_message)
        print('发送成功')
        # print(send_message)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False

    end_time = time.time()
    print(f'发送时间：{end_time - start_time}秒')

    return True





def try_connect(connect_ip: str, port_number: int, is_repeat: bool = False, max_reconnect_times: int = 50) -> (
                bool, socket.socket):
    """
    尝试连接.

    :param is_repeat: 是否是重新连接
    :param max_reconnect_times:最大重连次数
    :return: (连接状态True为成功, Socket / None)
    """
    reconnect_time = 0
    while reconnect_time < max_reconnect_times:
        logging.warning(f'尝试{"重新" if is_repeat else ""}发起第{reconnect_time + 1}次连接...')
        try:
            connected_sock = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
            connected_sock.connect((connect_ip, port_number))
        except Exception as e:
            reconnect_time += 1
            logging.error(f'第{reconnect_time}次连接失败... 5秒后重新连接...\n {e}')
            time.sleep(5)
            continue
        logging.warning(f'{"重新" if is_repeat else ""}连接成功')
        return True, connected_sock
    return False, None


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



#提取西红柿，使用S+L的图像
def extract_s_l(image):
    # image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    s_channel = hsv[:,:,1]
    l_channel = lab[:,:,0]
    result = cv2.add(s_channel, l_channel)
    return result

def find_reflection(image, threshold=190):
    # 读取图像
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 应用阈值分割
    _, reflection = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return reflection

def otsu_threshold(image):

    # 将图像转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Otsu阈值分割
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

# 提取花萼，使用G-R的图像
def extract_g_r(image):
    # image = cv2.imread(image_path)
    g_channel = image[:,:,1]
    r_channel = image[:,:,2]
    result = cv2.subtract(cv2.multiply(g_channel, 1.5), r_channel)
    return result


#提取西红柿，使用R-B的图像
def extract_r_b(image):
    # image = cv2.imread(image_path)
    r_channel = image[:,:,2]
    b_channel = image[:,:,0]
    result = cv2.subtract(r_channel, b_channel)
    return result

def extract_r_g(image):
    # image = cv2.imread(image_path)
    r_channel = image[:,:,2]
    g_channel = image[:,:,1]
    result = cv2.subtract(r_channel, g_channel)
    return result

def threshold_segmentation(image, threshold, color=255):
    _, result = cv2.threshold(image, threshold, color, cv2.THRESH_BINARY)
    return result

def bitwise_operation(image1, image2, operation='and'):
    if operation == 'and':
        result = cv2.bitwise_and(image1, image2)
    elif operation == 'or':
        result = cv2.bitwise_or(image1, image2)
    else:
        raise ValueError("operation must be 'and' or 'or'")
    return result

def largest_connected_component(bin_img):
    # 使用connectedComponentsWithStats函数找到连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    # 找到最大的连通区域（除了背景）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 创建一个新的二值图像，只显示最大的连通区域
    new_bin_img = np.zeros_like(bin_img)
    new_bin_img[labels == largest_label] = 255

    return new_bin_img

def close_operation(bin_img, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    return closed_img

def open_operation(bin_img, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return opened_img


def draw_tomato_edge(original_img, bin_img):
    bin_img_processed = close_operation(bin_img, kernel_size=(15, 15))
    # cv2.imshow('Close Operation', bin_img_processed)
    # bin_img_processed = open_operation(bin_img_processed, kernel_size=(19, 19))
    # cv2.imshow('Open Operation', bin_img_processed)
    # 现在使用处理后的bin_img_processed查找轮廓
    contours, _ = cv2.findContours(bin_img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，直接返回原图
    if not contours:
        return original_img
    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    # 多边形近似的精度调整
    epsilon = 0.0006 * cv2.arcLength(max_contour, True)  # 可以调整这个值
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # 绘制轮廓
    cv2.drawContours(original_img, [approx], -1, (0, 255, 0), 3)
    mask = np.zeros_like(bin_img)

    # 使用白色填充最大轮廓
    cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)

    return original_img, mask

def draw_tomato_edge_convex_hull(original_img, bin_img):
    bin_img_blurred = cv2.GaussianBlur(bin_img, (5, 5), 0)
    contours, _ = cv2.findContours(bin_img_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_img
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)
    cv2.drawContours(original_img, [hull], -1, (0, 255, 0), 3)
    return original_img

# 得到完整的西红柿二值图像，除了绿色花萼
def fill_holes(bin_img):
    # 复制 bin_img 到 img_filled
    img_filled = bin_img.copy()

    # 获取图像的高度和宽度
    height, width = bin_img.shape

    # 创建一个掩码，比输入图像大两个像素点
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # 使用 floodFill 函数填充黑色区域
    cv2.floodFill(img_filled, mask, (0, 0), 255)

    # 反转填充后的图像
    img_filled_d = cv2.bitwise_not(img_filled)

    # 使用 bitwise_or 操作合并原图像和填充后的图像
    img_filled = cv2.bitwise_or(bin_img, img_filled)
    # 裁剪 img_filled 和 img_filled_d 到与 bin_img 相同的大小
    # img_filled = img_filled[:height, :width]
    img_filled_d = img_filled_d[:height, :width]

    return img_filled, img_filled_d

def bitwise_and_rgb_with_binary(rgb_img, bin_img):
    # 将二值图像转换为三通道图像
    bin_img_3channel = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # 使用 bitwise_and 操作合并 RGB 图像和二值图像
    result = cv2.bitwise_and(rgb_img, bin_img_3channel)

    return result


def extract_max_connected_area(image, lower_hsv, upper_hsv):
    # 读取图像
    # image = cv2.imread(image_path)

    # 将图像从BGR转换到HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 使用阈值获取指定区域的二值图像
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 找到二值图像的连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 找到最大的连通区域（除了背景）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 创建一个新的二值图像，只显示最大的连通区域
    new_bin_img = np.zeros_like(mask)
    new_bin_img[labels == largest_label] = 255

    # 复制 new_bin_img 到 img_filled
    img_filled = new_bin_img.copy()

    # 获取图像的高度和宽度
    height, width = new_bin_img.shape

    # 创建一个掩码，比输入图像大两个像素点
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # 使用 floodFill 函数填充黑色区域
    cv2.floodFill(img_filled, mask, (0, 0), 255)

    # 反转填充后的图像
    img_filled_inv = cv2.bitwise_not(img_filled)

    # 使用 bitwise_or 操作合并原图像和填充后的图像
    img_filled = cv2.bitwise_or(new_bin_img, img_filled_inv)

    return img_filled
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

    return (max(major_axis, minor_axis), min(major_axis, minor_axis))

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
    return number_defects, total_pixels




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