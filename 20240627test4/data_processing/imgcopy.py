# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 下午5:11
# @Author  : TG
# @File    : imgcopy.py
# @Software: PyCharm
import os
import shutil

def copy_images_recursively(source_folder, target_folder):
    # 遍历源文件夹中的所有内容
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        if os.path.isdir(item_path):
            # 如果是文件夹，递归调用当前函数
            copy_images_recursively(item_path, target_folder)
        elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 如果是图片文件，则复制到目标文件夹
            shutil.copy(item_path, target_folder)
            print(f"Copied {item_path} to {target_folder}")

# 源文件夹路径
source_folder = r'D:\project\20240714Actual_deployed\20240718test\T'
# 目标文件夹路径
target_folder = r'D:\project\20240714Actual_deployed\20240718test\01img'

# 调用函数
copy_images_recursively(source_folder, target_folder)
