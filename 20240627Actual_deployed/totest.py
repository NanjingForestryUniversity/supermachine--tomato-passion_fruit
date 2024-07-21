# -*- coding: utf-8 -*-
# @Time    : 2024/7/7 下午4:33
# @Author  : TG
# @File    : totest.py
# @Software: PyCharm
import time

from detector import Detector_to
import numpy as np

import os
from PIL import Image




def main():
    s = []
    path = r'D:\project\20240627Actual_deployed\to'
    to = Detector_to()
    i = 1
    for filename in os.listdir(path):
        if filename.endswith('.bmp'):
            img_path = os.path.join(path, filename)
            image = Image.open(img_path)
            img = np.array(image)
            t = time.time()
            result = to.run(img)
            e = time.time()
            print(f'第{i}张图时间：{e-t}')
            print(f'图片名：{filename},结果：{result}')
            s.append(result)
            i += 1
    print(f'长度：{sum(s)}')


#0为褶皱，1为正常


if __name__ == '__main__':
    '''
    python与qt采用windows下的命名管道进行通信，数据流按照约定的通信协议进行
    数据处理逻辑为：连续接收5张RGB图，然后根据解析出的指令部分决定是否接收一张光谱图，然后进行处理，最后将处理得到的指标结果进行编码回传
    '''
    main()