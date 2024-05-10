# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 16:54
# @Author  : TG
# @File    : qt_test.py
# @Software: PyCharm

import numpy as np
import socket
import logging
import matplotlib.pyplot as plt
import cv2
import os
import time
from utils import DualSock, try_connect, receive_sock, parse_protocol, ack_sock, done_sock


def rec_socket(recv_sock: socket.socket, cmd_type: str, ack: bool) -> bool:
    if ack:
        cmd = 'A' + cmd_type
    else:
        cmd = 'D' + cmd_type
    while True:
        try:
            temp = recv_sock.recv(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return False
        except TimeoutError as e:
            logging.error(f'超时了，错误代码: \n{e}')
            return False
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return False
        if temp == b'\xaa':
            break

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        try:
            temp += recv_sock.recv(1)
        except Exception as e:
            logging.error(f'接收报文长度失败, 错误代码: \n{e}')
            return False
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}, \n报文内容\n{temp}')
        return False

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        try:
            temp += recv_sock.recv(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}，\n报文内容\n{temp}')
            return False
    data = temp
    if cmd.strip().upper() != data[:4].decode('ascii').strip().upper():
        logging.error(f'客户端接收指令错误,\n指令内容\n{data}')
        return False
    else:
        if cmd == 'DIM':
            print(data)

        # 进行数据校验
        temp = b''
        while len(temp) < 3:
            try:
                temp += recv_sock.recv(1)
            except Exception as e:
                logging.error(f'接收报文校验失败, 错误代码: \n{e}')
                return False
        if temp == b'\xff\xff\xbb':
            return True
        else:
            logging.error(f"接收了一个完美的只错了校验位的报文，\n data: {data}")
            return False


# def main():
#     socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     socket_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     socket_receive.bind(('127.0.0.1', 21123))
#     socket_receive.listen(5)
#     socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     socket_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     socket_send.bind(('127.0.0.1', 21122))
#     socket_send.listen(5)
#     print('等待连接')
#     socket_send_1, receive_addr_1 = socket_send.accept()
#     print("连接成功：", receive_addr_1)
#     # socket_send_2 = socket_send_1
#     socket_send_2, receive_addr_2 = socket_receive.accept()
#     print("连接成功：", receive_addr_2)
#     while True:
#         cmd = input('请输入指令：').strip().upper()
#         if cmd == 'IM':
#             with open('data/newrawfile_ref.raw', 'rb') as f:
#                 data = np.frombuffer(f.read(), dtype=np.float32).reshape(750, 288, 384)
#             data = data[:, [91, 92, 93, 94, 95, 96, 97, 98, 99, 100], :]
#             n_rows, n_bands, n_cols = data.shape[0], data.shape[1], data.shape[2]
#             print(f'n_rows：{n_rows}, n_bands：{n_bands}, n_cols：{n_cols}')
#             n_rows, n_cols, n_bands = [x.to_bytes(2, byteorder='big') for x in [n_rows, n_cols, n_bands]]
#             data = data.tobytes()
#             length = len(data) + 10
#             print(f'length: {length}')
#             length = length.to_bytes(4, byteorder='big')
#             msg = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + n_rows + n_cols + n_bands + data + b'\xff\xff\xbb'
#             socket_send_1.send(msg)
#             print('发送成功')
#             result = socket_send_2.recv(5)
#             length = int.from_bytes(result[1:5], byteorder='big')
#             result = b''
#             while len(result) < length:
#                 result += socket_send_2.recv(length)
#             print(result)
#             data = result[4:length].decode()
#             print(data)

def main():
    socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_receive.bind(('127.0.0.1', 21123))
    socket_receive.listen(5)
    socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_send.bind(('127.0.0.1', 21122))
    socket_send.listen(5)
    print('等待连接')
    socket_send_1, receive_addr_1 = socket_send.accept()
    print("连接成功：", receive_addr_1)
    socket_send_2, receive_addr_2 = socket_receive.accept()
    print("连接成功：", receive_addr_2)

    # while True:
    #     cmd = input().strip().upper()
    #     if cmd == 'IM':
    #         image_dir = r'D:\project\Tomato\20240410tomatoRGBtest2\data'
    #         image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".bmp")]
    #         for image_path in image_paths:
    #             img = cv2.imread(image_path)
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             img = np.asarray(img, dtype=np.uint8)
    #             width = img.shape[0]
    #             height = img.shape[1]
    #             print(width, height)
    #             img_bytes = img.tobytes()
    #             length = len(img_bytes) + 8
    #             print(length)
    #             length = length.to_bytes(4, byteorder='big')
    #             width = width.to_bytes(2, byteorder='big')
    #             height = height.to_bytes(2, byteorder='big')
    #             send_message = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + width + height + img_bytes + b'\xff\xff\xbb'
    #             socket_send_1.send(send_message)
    #             print('发送成功')
    #     result = socket_send_2.recv(5)
    #     print(result)
    while True:
        cmd = input().strip().upper()
        if cmd == 'IM':
            image_dir = r'D:\project\Tomato\20240410tomatoRGBtest2\data'
            send_images(image_dir, socket_send_1, socket_send_2)
def send_images(image_dir, socket_send_1, socket_send_2):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".bmp")]
    num_images = len(image_paths)
    num_groups = (num_images + 4) // 5

    for group_idx in range(num_groups):
        start = group_idx * 5
        end = start + 5
        group_images = image_paths[start:end]

        group_start = time.time()

        for image_path in group_images:

            img_start = time.time()

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, dtype=np.uint8)
            width = img.shape[0]
            height = img.shape[1]
            print(width, height)
            img_bytes = img.tobytes()
            length = len(img_bytes) + 8
            print(length)
            length = length.to_bytes(4, byteorder='big')
            width = width.to_bytes(2, byteorder='big')
            height = height.to_bytes(2, byteorder='big')
            send_message = b'\xaa' + length + ('  ' + 'IM').upper().encode('ascii') + width + height + img_bytes + b'\xff\xff\xbb'
            socket_send_1.send(send_message)

            img_end = time.time()
            print(f'图片发送时间: {img_end - img_start}秒')

            print('图片发送成功')

        group_end = time.time()
        print(f'第 {group_idx + 1} 组图片发送时间: {group_end - group_start}秒')

        result = socket_send_2.recv(5)
        print(f'第 {group_idx + 1} 组结果: {result}')




if __name__ == '__main__':
    main()