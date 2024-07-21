# -*- coding: utf-8 -*-
# @Time    : 2024/7/16 下午8:58
# @Author  : GG
# @File    : pf_zz_test.py
# @Software: PyCharm
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
from classifer import ImageClassifier
from config import Config as setting


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def natural_sort_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


    # "0": "De" 褶皱
    # "1": "N"  正常


def main():
    image_dir = r'D:\project\20240714Actual_deployed\zz_test\TEST'
    pf_zz = ImageClassifier(model_path=setting.imgclassifier_model_path,
                            class_indices_path=setting.imgclassifier_class_indices_path) # 假设 TOSEG 是已定义好的类，可以处理图片分割
    # 获取所有.bmp文件，并进行自然排序
    rgb_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.bmp')]
    rgb_files.sort(key=natural_sort_key)

    # 准备保存到 Excel 的数据
    records = []

    for idx, image_path in enumerate(rgb_files):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t = time.time()
        result = pf_zz.predict(img)
        e = time.time()

        process_time = (e - t) * 1000
        print(f'第{idx + 1}张图时间：{process_time}')
        print(f'结果：{result}')

    ## 控制台显示识别结果
    #     records.append(result)
    # print(f'识别为正常未褶皱的数量：{sum(records)}')


    ## 将结果及原始文件信息写入excel
        # 获取原始文件名
        original_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 添加记录到列表
        records.append({
            "图片序号": idx + 1,
            "图片名": original_filename,
            "识别结果(0为褶皱，1为正常）": result,
            "处理时间(ms)": process_time
        })

    # 创建 DataFrame 并写入 Excel 文件
    df = pd.DataFrame(records)
    df.to_excel(r'./zz_result.xlsx', index=False)




if __name__ == '__main__':
    main()


