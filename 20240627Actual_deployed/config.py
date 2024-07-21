# -*- coding: utf-8 -*-
# @Time    : 2024/6/17 下午3:36
# @Author  : TG
# @File    : config.py
# @Software: PyCharm

from root_dir import ROOT_DIR

class Config:
    #文件相关参数
    #预热参数
    n_spec_rows, n_spec_cols, n_spec_bands = 25, 30, 13
    n_rgb_rows, n_rgb_cols, n_rgb_bands = 613, 800, 3
    tomato_img_dir = ROOT_DIR / 'models' / 'TO.bmp'
    passion_fruit_img_dir = ROOT_DIR / 'models' / 'PF.bmp'
    #模型路径
    #糖度模型
    brix_model_path = ROOT_DIR / 'models' / 'passion_fruit.joblib'
    #图像分类模型
    imgclassifier_model_path = ROOT_DIR / 'models' / 'resnet18pf20240705.pth'
    imgclassifier_class_indices_path = ROOT_DIR / 'models' / 'class_indices.json'

    #番茄破损模型
    tomato_model_path = ROOT_DIR / 'weights' / 'best.pt'

    #classifer.py参数
    #tomato
    find_reflection_threshold = 190
    extract_g_r_factor = 1.5

    #passion_fruit
    hue_value = 37
    hue_delta = 10
    value_target = 25
    value_delta = 10

    #提取绿色像素参数
    low_H = 0
    low_S = 100
    low_V = 0
    high_H = 60
    high_S = 180
    high_V = 60

    #spec_predict
    #筛选谱段并未使用，在qt取数据时已经筛选
    selected_bands = [8, 9, 10, 48, 49, 50, 77, 80, 103, 108, 115, 143, 145]

    #data_processing
    #根据标定数据计算的参数,实际长度/像素长度，单位cm
    pixel_length_ratio = 6.3/425
    #绿叶面积阈值，高于此阈值认为连通域是绿叶
    area_threshold = 20000
    #百香果密度（g/cm^3）
    density = 0.652228972
    #百香果面积比例，每个像素代表的实际面积（cm^2）
    area_ratio = 0.00021973702422145334

    #def analyze_tomato
    #s_l通道阈值
    threshold_s_l = 180
    threshold_fore_g_r_t = 20

