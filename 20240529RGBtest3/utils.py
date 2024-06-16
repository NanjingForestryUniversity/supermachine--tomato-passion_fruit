# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 18:24
# @Author  : TG
# @File    : utils.py
# @Software: PyCharm


import shutil

import os



import win32file
import win32pipe
import time
import logging
import numpy as np
from PIL import Image
import io

class Pipe:
    def __init__(self, rgb_receive_name, rgb_send_name, spec_receive_name):
        self.rgb_receive_name = rgb_receive_name
        self.rgb_send_name = rgb_send_name
        self.spec_receive_name = spec_receive_name
        self.rgb_receive = None
        self.rgb_send = None
        self.spec_receive = None

    def create_pipes(self, rgb_receive_name, rgb_send_name, spec_receive_name):
        while True:
            try:
                # 打开或创建命名管道
                self.rgb_receive = win32pipe.CreateNamedPipe(
                    rgb_receive_name,
                    win32pipe.PIPE_ACCESS_INBOUND,
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                    1, 80000000, 80000000, 0, None
                )
                self.rgb_send = win32pipe.CreateNamedPipe(
                    rgb_send_name,
                    win32pipe.PIPE_ACCESS_OUTBOUND,  # 修改为输出模式
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                    1, 80000000, 80000000, 0, None
                )
                self.spec_receive = win32pipe.CreateNamedPipe(
                    spec_receive_name,
                    win32pipe.PIPE_ACCESS_INBOUND,
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
                    1, 200000000, 200000000, 0, None
                )
                print("pipe管道创建成功，等待连接...")
                # 等待发送端连接
                win32pipe.ConnectNamedPipe(self.rgb_receive, None)
                print("rgb_receive connected.")
                # 等待发送端连接
                win32pipe.ConnectNamedPipe(self.rgb_send, None)
                print("rgb_send connected.")
                win32pipe.ConnectNamedPipe(self.spec_receive, None)
                print("spec_receive connected.")
                return self.rgb_receive, self.rgb_send, self.spec_receive

            except Exception as e:
                print(f"管道创建连接失败，失败原因: {e}")
                print("等待5秒后重试...")
                time.sleep(5)
                continue

    def receive_rgb_data(self, rgb_receive):
        try:
            # 读取图片数据长度
            len_img = win32file.ReadFile(rgb_receive, 4, None)
            data_size = int.from_bytes(len_img[1], byteorder='big')
            # 读取实际图片数据
            result, data = win32file.ReadFile(rgb_receive, data_size, None)
            # 检查读取操作是否成功
            if result != 0:
                print(f"读取失败，错误代码: {result}")
                return None
            # 返回成功读取的数据
            return data
        except Exception as e:
            print(f"数据接收失败，错误原因: {e}")
            return None

    def receive_spec_data(self, spec_receive):
        try:
            # 读取光谱数据长度
            len_spec = win32file.ReadFile(spec_receive, 4, None)
            data_size = int.from_bytes(len_spec[1], byteorder='big')
            # 读取光谱数据
            result, spec_data = win32file.ReadFile(spec_receive, data_size, None)
            # 检查读取操作是否成功
            if result != 0:
                print(f"读取失败，错误代码: {result}")
                return None
            # 返回成功读取的数据
            return spec_data
        except Exception as e:
            print(f"数据接收失败，错误原因: {e}")
            return None

    def parse_img(self, data: bytes) -> (str, any):
        """
        图像数据转换.

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

        n_rows, n_cols, img = data[:2], data[2:4], data[4:]
        try:
            n_rows, n_cols = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols:{n_cols}')
            return '', None
        try:
            assert n_rows * n_cols * 3 == len(img)
            # 因为是float32类型 所以长度要乘12 ，如果是uint8则乘3
        except AssertionError:
            logging.error('图像指令IM转换失败，数据长度错误')
            return '', None
        img = np.frombuffer(img, dtype=np.uint8).reshape(n_rows, n_cols, -1)
        return cmd, img

    def parse_spec(self, data: bytes) -> (str, any):
        """
        光谱数据转换.

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

        n_rows, n_cols, n_bands, spec = data[:2], data[2:4], data[4:6], data[6:]
        try:
            n_rows, n_cols, n_bands = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols, n_bands]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols:{n_cols}, n_bands:{n_bands}')
            return '', None
        try:
            assert n_rows * n_cols * n_bands * 2 == len(spec)

        except AssertionError:
            logging.error('图像指令转换失败，数据长度错误')
            return '', None
        spec = np.frombuffer(spec, dtype=np.uint16).reshape((n_rows, n_bands, -1)).transpose(0, 2, 1)
        return cmd, spec

    def send_data(self,cmd:str, brix, green_percentage, weigth, diameter, defect_num, total_defect_area, rp):
        # start_time = time.time()
        #
        # rp1 = Image.fromarray(rp.astype(np.uint8))
        # # cv2.imwrite('rp1.bmp', rp1)
        #
        # # 将 Image 对象保存到 BytesIO 流中
        # img_bytes = io.BytesIO()
        # rp1.save(img_bytes, format='BMP')
        # img_bytes = img_bytes.getvalue()

        # width = rp.shape[0]
        # height = rp.shape[1]
        # print(width, height)
        # img_bytes = rp.tobytes()
        # length = len(img_bytes) + 18
        # print(length)
        # length = length.to_bytes(4, byteorder='big')
        # width = width.to_bytes(2, byteorder='big')
        # height = height.to_bytes(2, byteorder='big')
        cmd = cmd.strip().upper()
        cmd_type = 'RE'
        cmd_re = cmd_type.upper().encode('ascii')
        img = np.asarray(rp, dtype=np.uint8)  # 将图像转换为 NumPy 数组
        height = img.shape[0]  # 获取图像的高度
        width = img.shape[1]  # 获取图像的宽度
        height = height.to_bytes(2, byteorder='big')
        width = width.to_bytes(2, byteorder='big')
        img_bytes = img.tobytes()
        diameter = diameter.to_bytes(2, byteorder='big')
        defect_num = defect_num.to_bytes(2, byteorder='big')
        total_defect_area = int(total_defect_area).to_bytes(4, byteorder='big')
        length = len(img_bytes) + 18
        length = length.to_bytes(4, byteorder='big')
        if cmd == 'TO':
            brix = 0
            brix = brix.to_bytes(2, byteorder='big')
            gp = green_percentage.to_bytes(1, byteorder='big')
            weigth = 0
            weigth = weigth.to_bytes(1, byteorder='big')
            send_message = length + cmd_re + brix + gp + diameter + weigth + defect_num + total_defect_area + height + width + img_bytes
        elif cmd == 'PF':
            brix = int(brix.item() * 1000).to_bytes(2, byteorder='big')
            gp = 0
            gp = gp.to_bytes(1, byteorder='big')
            weigth = weigth.to_bytes(1, byteorder='big')
            send_message = length + cmd_re + brix + gp + diameter + weigth + defect_num + total_defect_area + height + width + img_bytes
        try:
            win32file.WriteFile(self.rgb_send, send_message)
            time.sleep(0.01)
            print('发送成功')
            print(len(send_message), len(img_bytes))
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