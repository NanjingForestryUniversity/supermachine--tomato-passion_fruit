# -*- coding: utf-8 -*-
# @Time    : 2024/7/7 下午4:33
# @Author  : TG
# @File    : totest.py
# @Software: PyCharm
import time
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
from to_seg import TOSEG


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def natural_sort_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def main():
    image_dir = r'D:\project\20240714Actual_deployed\testimg'
    to = TOSEG()  # 假设 TOSEG 是已定义好的类，可以处理图片分割
    # 获取所有.bmp文件，并进行自然排序
    rgb_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.bmp')]
    rgb_files.sort(key=natural_sort_key)

    # 准备保存到 Excel 的数据
    records = []

    for idx, image_path in enumerate(rgb_files):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = time.time()
        result = to.toseg(img)  # 假设 toseg 方法接受一个图片数组，并返回处理后的图片
        e = time.time()
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        process_time = e - t
        print(f'第{idx + 1}张图时间：{process_time}')

        # 获取原始文件名并添加“mask”后缀
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f'{original_filename}_leaf.png'
        cv2.imwrite(os.path.join(r'D:\project\20240714Actual_deployed\leaf',
                                 output_filename), result)

        # 添加记录到列表
        records.append({
            "Image Index": idx + 1,
            "File Name": original_filename,
            "Processing Time (s)": process_time
        })

        # cv2.imshow('result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 创建 DataFrame 并写入 Excel 文件
    df = pd.DataFrame(records)
    df.to_excel(r'D:\project\20240714Actual_deployed\leaf\leaf_processing_times.xlsx', index=False)


if __name__ == '__main__':
    main()


