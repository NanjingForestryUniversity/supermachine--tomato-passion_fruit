# -*- coding: utf-8 -*-
# @Time    : 2024/7/4 下午10:43
# @Author  : TG
# @File    : pic.py
# @Software: PyCharm
import os


def rename_bmp_images(folder_path, new_name_format):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 过滤出BMP图像文件
    bmp_files = [f for f in files if f.lower().endswith('.bmp')]

    # 对每个BMP图像文件进行重命名
    for index, bmp_file in enumerate(bmp_files):
        old_path = os.path.join(folder_path, bmp_file)
        new_name = new_name_format.format(index + 1)
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')


# 指定文件夹路径和新的命名格式
folder_path = r'D:\桌面文件\裂口数据集扩充（4月份数据补充）\scar'
new_name_format = 'scar_{:03d}.bmp'  # 例如，image_001.bmp, image_002.bmp, ...

# 调用函数进行重命名
rename_bmp_images(folder_path, new_name_format)
