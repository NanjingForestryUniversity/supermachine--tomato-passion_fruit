# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 18:45
# @Author  : TG
# @File    : main.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 15:04
# @Author  : TG
# @File    : main.py
# @Software: PyCharm

import socket
import sys
import numpy as np
import cv2
import root_dir
import time
import os
from root_dir import ROOT_DIR
import logging
from utils import create_pipes, receive_rgb_data, send_data, receive_spec_data, analyze_tomato
from collections import deque
import time
import io
from PIL import Image
import threading
import queue

def process_data(img: any) -> tuple:
    """
    处理指令

    :param cmd: 指令类型
    :param data: 指令内容
    :param connected_sock: socket
    :param detector: 模型
    :return: 是否处理成功
    """
    # if cmd == 'IM':

    long_axis, short_axis, number_defects, total_pixels, rp = analyze_tomato(img)
    return long_axis, short_axis, number_defects, total_pixels, rp


## 20240423代码
def main(is_debug=False):
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'report.log'))
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler],
                        level=logging.DEBUG)
    rgb_receive_name = r'\\.\pipe\rgb_receive'
    rgb_send_name = r'\\.\pipe\rgb_send'
    spec_receive_name = r'\\.\pipe\spec_receive'
    rgb_receive, rgb_send, spec_receive = create_pipes(rgb_receive_name, rgb_send_name, spec_receive_name)

    # data_size = 15040566

    while True:
        long_axis_list = []
        short_axis_list = []
        defect_num_sum = 0
        total_defect_area_sum = 0
        rp = None

        start_time = time.time()

        for i in range(5):

            # start_time = time.time()


            img_data = receive_rgb_data(rgb_receive)
            image = Image.open(io.BytesIO(img_data))
            img = np.array(image)
            print(img.shape)

            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f'接收时间：{elapsed_time}秒')

            long_axis, short_axis, number_defects, total_pixels, rp = process_data(img=img)
            # print(long_axis, short_axis, number_defects, type(total_pixels), rp.shape)

            if i <= 2:
                long_axis_list.append(long_axis)
                short_axis_list.append(short_axis)
            if i == 1:
                rp_result = rp

            defect_num_sum += number_defects
            total_defect_area_sum += total_pixels

            long_axis = round(sum(long_axis_list) / 3)
            short_axis = round(sum(short_axis_list) / 3)
        # print(type(long_axis), type(short_axis), type(defect_num_sum), type(total_defect_area_sum), type(rp_result))

        spec_data = receive_spec_data(spec_receive)
        print(f'光谱数据接收长度：', len(spec_data))


        response = send_data(pipe_send=rgb_send, long_axis=long_axis, short_axis=short_axis,
                             defect_num=defect_num_sum, total_defect_area=total_defect_area_sum, rp=rp_result)



        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f'总时间：{elapsed_time}毫秒')

        print(long_axis, short_axis, defect_num_sum, total_defect_area_sum, rp_result.shape)



if __name__ == '__main__':
    # 2个pipe管道
    # 接收到图片 n_rows * n_cols * 3， uint8
    # 发送long_axis, short_axis, defect_num_sum, total_defect_area_sum, rp_result
    main(is_debug=False)




