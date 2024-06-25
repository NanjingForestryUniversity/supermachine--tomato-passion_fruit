# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 18:45
# @Author  : TG
# @File    : main.py
# @Software: PyCharm

import sys
import os
import cv2
from root_dir import ROOT_DIR
from classifer import Spec_predict, Data_processing
# from classifer import ImageClassifier
import logging
from utils import Pipe
import numpy as np
import time




def process_data(cmd: str, images: list, spec: any, dp: Data_processing, pipe: Pipe, detector: Spec_predict) -> bool:
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
                gp = round(green_percentage, 2)

        elif cmd == 'PF':
            # 百香果
            diameter, weight, number_defects, total_pixels, rp = dp.analyze_passion_fruit(img)
            if i <= 2:
                diameter_axis_list.append(diameter)
                max_defect_num = max(max_defect_num, number_defects)
                max_total_defect_area = max(max_total_defect_area, total_pixels)
            if i == 1:
                rp_result = rp
                weight = weight

        else:
            logging.error(f'错误指令，指令为{cmd}')
            return False

    diameter = round(sum(diameter_axis_list) / 3, 2)

    if cmd == 'TO':
        brix = 0
        weight = 0
        # print(f'预测的brix值为：{brix}; 预测的直径为：{diameter}; 预测的重量为：{weight}; 预测的绿色比例为：{gp};'
        #       f' 预测的缺陷数量为：{max_defect_num}; 预测的总缺陷面积为：{max_total_defect_area};')
        response = pipe.send_data(cmd=cmd, brix=brix, diameter=diameter, green_percentage=gp, weight=weight,
                                  defect_num=max_defect_num, total_defect_area=max_total_defect_area, rp=rp_result)
        return response
    elif cmd == 'PF':
        green_percentage = 0
        brix = detector.predict(spec)
        if diameter == 0:
            brix = 0
        # print(f'预测的brix值为：{brix}; 预测的直径为：{diameter}; 预测的重量为：{weight}; 预测的绿色比例为：{green_percentage};'
        #       f' 预测的缺陷数量为：{max_defect_num}; 预测的总缺陷面积为：{max_total_defect_area};')
        response = pipe.send_data(cmd=cmd, brix=brix, green_percentage=green_percentage, diameter=diameter, weight=weight,
                                  defect_num=max_defect_num, total_defect_area=max_total_defect_area, rp=rp_result)
        return response

def main(is_debug=False):
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'tomato.log'), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler],
                        level=logging.DEBUG)
    #模型加载
    detector = Spec_predict(ROOT_DIR/'models'/'passion_fruit_2.joblib')
    # classifier = ImageClassifier(ROOT_DIR/'models'/'resnet34_0619.pth', ROOT_DIR/'models'/'class_indices.json')
    dp = Data_processing()
    print('系统初始化中...')
    #模型预热
    _ = detector.predict(np.ones((30, 30, 224), dtype=np.uint16))
    # _ = classifier.predict(np.ones((224, 224, 3), dtype=np.uint8))
    # _, _, _, _, _ =dp.analyze_tomato(cv2.imread(r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\data\tomato_img\bad\71.bmp'))
    # _, _, _, _, _ = dp.analyze_passion_fruit(cv2.imread(r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\data\passion_fruit_img\38.bmp'))
    print('系统初始化完成')

    rgb_receive_name = r'\\.\pipe\rgb_receive'
    rgb_send_name = r'\\.\pipe\rgb_send'
    spec_receive_name = r'\\.\pipe\spec_receive'
    pipe = Pipe(rgb_receive_name, rgb_send_name, spec_receive_name)
    rgb_receive, rgb_send, spec_receive = pipe.create_pipes(rgb_receive_name, rgb_send_name, spec_receive_name)
    # 预热循环，只处理cmd为'YR'的数据
    # 当接收到的第一个指令预热命令时，结束预热循环
    while True:
        # start_time00 = time.time()
        data = pipe.receive_rgb_data(rgb_receive)
        cmd, _ = pipe.parse_img(data)
        # end_time00 = time.time()
        # print(f'接收预热数据时间：{(end_time00 - start_time00) * 1000}毫秒')
        if cmd == 'YR':
            break
    #主循环
    # q = 1
    while True:
        #RGB图像部分
        # start_time = time.time()
        images = []
        cmd = None
        for _ in range(5):
            # start_time1 = time.time()
            data = pipe.receive_rgb_data(rgb_receive)
            # end_time10 = time.time()
            # print(f'接收第{q}组第{i}份RGB数据时间：{(end_time10 - start_time1) * 1000}毫秒')

            # start_time11 = time.time()
            cmd, img = pipe.parse_img(data)
            # end_time1 = time.time()
            # print(f'解析第{q}组第{i}份RGB数据时间：{(end_time1 - start_time11) * 1000}毫秒')
            # print(f'接收第{q}组第{i}张RGB图时间：{(end_time1 - start_time1) * 1000}毫秒')

            # 使用分类器进行预测
            # prediction = classifier.predict(img)
            # print(f'预测结果：{prediction}')
            #默认全为有果
            prediction = 1
            if prediction == 1:
                images.append(img)
            else:
                response = pipe.send_data(cmd='KO', brix=0, diameter=0, green_percentage=0, weigth=0, defect_num=0,
                                           total_defect_area=0, rp=np.zeros((100, 100, 3), dtype=np.uint8))
                logging.info("图像中无果，跳过此图像")
                continue

        if cmd not in ['TO', 'PF', 'YR', 'KO']:
            logging.error(f'错误指令，指令为{cmd}')
            continue
        #Spec数据部分
        spec = None
        if cmd == 'PF':
            # start_time2 = time.time()
            spec_data = pipe.receive_spec_data(spec_receive)
            # print(f'接收第{q}组光谱数据长度：{len(spec_data)}')
            _, spec = pipe.parse_spec(spec_data)
            # print(f'处理第{q}组光谱数据长度：{len(spec)}')
            # print(spec.shape)
            # print(f'解析第{q}组光谱数据时间：{(time.time() - start_time2) * 1000}毫秒')
            # end_time2 = time.time()
            # print(f'接收第{q}组光谱数据时间：{(end_time2 - start_time2) * 1000}毫秒')
        #数据处理部分
        # start_time3 = time.time()
        if images:  # 确保images不为空
            response = process_data(cmd, images, spec, dp, pipe, detector)
            end_time3 = time.time()
            # print(f'第{q}组处理时间：{(end_time3 - start_time3) * 1000}毫秒')
            if response:
                logging.info(f'处理成功，响应为: {response}')
            else:
                logging.error('处理失败')
        else:
            logging.error("没有有效的图像进行处理")

        # end_time = time.time()
        # print(f'第{q}组全流程时间：{(end_time - start_time) * 1000}毫秒')
        # q += 1


if __name__ == '__main__':
    '''
    python与qt采用windows下的命名管道进行通信，数据流按照约定的通信协议进行
    数据处理逻辑为：连续接收5张RGB图，然后根据解析出的指令部分决定是否接收一张光谱图，然后进行处理，最后将处理得到的指标结果进行编码回传
    '''
    main(is_debug=False)
