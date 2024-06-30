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
from config import Config
import time

def main(is_debug=False):
    setting = Config()
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'tomato-passion_fruit.log'), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler],
                        level=logging.DEBUG)
    #模型加载
    detector = Spec_predict()
    detector.load(path=setting.brix_model_path)
    dp = Data_processing()
    print('系统初始化中...')
    #模型预热
    #与qt_test测试时需要注释掉预热，模型接收尺寸为（25，30，13），qt_test发送的数据为（30，30，224），需要对数据进行切片（classifer.py第379行）
    _ = detector.predict(np.ones((setting.n_spec_rows, setting.n_spec_cols, setting.n_spec_bands), dtype=np.uint16))
    time.sleep(1)
    print('系统初始化完成')

    rgb_receive_name = r'\\.\pipe\rgb_receive'
    rgb_send_name = r'\\.\pipe\rgb_send'
    spec_receive_name = r'\\.\pipe\spec_receive'
    pipe = Pipe(rgb_receive_name, rgb_send_name, spec_receive_name)
    rgb_receive, rgb_send, spec_receive = pipe.create_pipes(rgb_receive_name, rgb_send_name, spec_receive_name)
    # 预热循环，只处理cmd为'YR'的数据
    # 当接收到的第一个指令预热命令时，结束预热循环
    while True:
        data = pipe.receive_rgb_data(rgb_receive)
        cmd, _ = pipe.parse_img(data)
        if cmd == 'YR':
            break
    #主循环
    q = 1
    while True:
        #RGB图像部分
        images = []
        cmd = None
        for _ in range(5):
            data = pipe.receive_rgb_data(rgb_receive)
            cmd, img = pipe.parse_img(data)
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
            spec_data = pipe.receive_spec_data(spec_receive)
            _, spec = pipe.parse_spec(spec_data)
        #数据处理部分
        if images:  # 确保images不为空
            response = dp.process_data(cmd, images, spec, pipe, detector)
            if response:
                logging.info(f'处理成功，响应为: {response}')
            else:
                logging.error('处理失败')
        else:
            logging.error("没有有效的图像进行处理")
        print(f'第{q}个果子处理完成')
        q += 1


if __name__ == '__main__':
    '''
    python与qt采用windows下的命名管道进行通信，数据流按照约定的通信协议进行
    数据处理逻辑为：连续接收5张RGB图，然后根据解析出的指令部分决定是否接收一张光谱图，然后进行处理，最后将处理得到的指标结果进行编码回传
    '''
    main(is_debug=False)