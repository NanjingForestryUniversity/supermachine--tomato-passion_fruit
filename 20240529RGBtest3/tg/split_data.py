# -*- coding: utf-8 -*-
# @Time    : 2024/7/11 下午3:08
# @Author  : TG
# @File    : split_data.py
# @Software: PyCharm
import os
import random
import shutil
from pathlib import Path

# 设置数据集目录
dataset_path = Path(r'F:\0711_lk')
images = list(dataset_path.glob('*.bmp'))  # 假设图像文件是jpg格式

# 设置随机种子以保证结果可复现
random.seed(42)

# 打乱数据集
random.shuffle(images)

# 计算划分点
num_images = len(images)
train_split = int(num_images * 0.6)
val_split = int(num_images * 0.8)

# 分割数据集
train_images = images[:train_split]
val_images = images[train_split:val_split]
test_images = images[val_split:]

# 创建保存分割后数据集的文件夹
(train_path, val_path, test_path) = [dataset_path.parent / x for x in ['train', 'val', 'test']]
for path in [train_path, val_path, test_path]:
    path.mkdir(exist_ok=True)

# 定义一个函数来复制图像和标签文件
def copy_files(files, dest_folder):
    for file in files:
        shutil.copy(file, dest_folder)
        label_file = file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy(label_file, dest_folder)

# 复制文件到新的文件夹
copy_files(train_images, train_path)
copy_files(val_images, val_path)
copy_files(test_images, test_path)

print("数据集划分完成。训练集、验证集和测试集已经被保存到对应的文件夹。")
