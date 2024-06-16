# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 18:45
# @Author  : TG
# @File    : main.py
# @Software: PyCharm

import sys
import os
from root_dir import ROOT_DIR
from classifer import Spec_predict, Data_processing
import logging
from utils import Pipe
import numpy as np


rgb_receive_name = r'\\.\pipe\rgb_receive'
rgb_send_name = r'\\.\pipe\rgb_send'
spec_receive_name = r'\\.\pipe\spec_receive'
pipe = Pipe(rgb_receive_name, rgb_send_name, spec_receive_name)
dp = Data_processing()
rgb_receive, rgb_send, spec_receive = pipe.create_pipes(rgb_receive_name, rgb_send_name, spec_receive_name)

def process_data(cmd: str, images: list, spec: any, detector: Spec_predict) -> bool:
    """
    处理指令

    :param cmd: 指令类型
    :param images: 图像数据列表
    :param spec: 光谱数据
    :param detector: 模型
    :return: 是否处理成功
    """
    diameter_axis_list = []
    max_defect_num = 0  # 初始化最大缺陷数量为0
    max_total_defect_area = 0  # 初始化最大总像素数为0

    for i, img in enumerate(images):
        if cmd == 'TO':
            # 番茄
            diameter, green_percentage, number_defects, total_pixels, rp = dp.analyze_tomato(img)
            if i <= 2:
                diameter_axis_list.append(diameter)
                max_defect_num = max(max_defect_num, number_defects)
                max_total_defect_area = max(max_total_defect_area, total_pixels)
            if i == 1:
                rp_result = rp
                gp = green_percentage

        elif cmd == 'PF':
            # 百香果
            diameter, weigth, number_defects, total_pixels, rp = dp.analyze_passion_fruit(img)
            if i <= 2:
                diameter_axis_list.append(diameter)
                max_defect_num = max(max_defect_num, number_defects)
                max_total_defect_area = max(max_total_defect_area, total_pixels)
            if i == 1:
                rp_result = rp
                weigth = weigth

        else:
            logging.error(f'错误指令，指令为{cmd}')
            return False

    diameter = round(sum(diameter_axis_list) / 3)

    if cmd == 'TO':
        response = pipe.send_data(cmd=cmd, diameter=diameter, green_percentage=gp,
                                  defect_num=max_defect_num, total_defect_area=max_total_defect_area, rp=rp_result)
    elif cmd == 'PF':
        brix = detector.predict(spec)
        response = pipe.send_data(cmd=cmd, brix=brix, diameter=diameter, weigth=weigth,
                                  defect_num=max_defect_num, total_defect_area=max_total_defect_area, rp=rp_result)
    return response

def main(is_debug=False):
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'tomato.log'))
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levellevel)s - %(message)s',
                        handlers=[file_handler, console_handler],
                        level=logging.DEBUG)
    detector = Spec_predict(ROOT_DIR/'20240529RGBtest3'/'models'/'passion_fruit.joblib')

    while True:
        images = []
        cmd = None

        for _ in range(5):
            data = pipe.receive_rgb_data(rgb_receive)
            cmd, img = pipe.parse_img(data)
            images.append(img)

        if cmd not in ['TO', 'PF']:
            logging.error(f'错误指令，指令为{cmd}')
            continue

        spec = None
        if cmd == 'PF':
            spec_data = pipe.receive_spec_data(spec_receive)
            _, spec = pipe.parse_spec(spec_data)

        response = process_data(cmd, images, spec, detector)
        if response:
            logging.info(f'处理成功，响应为: {response}')
        else:
            logging.error('处理失败')

if __name__ == '__main__':
    main(is_debug=False)
