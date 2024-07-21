import os
import re

def natural_sort_key(s):
    """提取文本中的数字作为排序键"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def rename_bmp_images(folder_path, prefix, suffix):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 过滤出BMP图像文件并进行自然排序
    bmp_files = sorted([f for f in files if f.lower().endswith('.bmp')], key=natural_sort_key)

    # 对每个BMP图像文件进行重命名
    for index, bmp_file in enumerate(bmp_files):
        old_path = os.path.join(folder_path, bmp_file)
        # 格式化新文件名，例如：1-1-1.bmp, 1-2-1.bmp, ...
        new_name = f"{prefix}-{index + 1}-{suffix}.bmp"
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')

# 指定文件夹路径
folder_path = r'D:\project\20240714Actual_deployed\20240718test\T\bottom'
folder_path1 = r'D:\project\20240714Actual_deployed\20240718test\T\middle'
folder_path2 = r'D:\project\20240714Actual_deployed\20240718test\T\top'

num = '1'
# 调用函数进行重命名
rename_bmp_images(folder_path, prefix=num, suffix='1')
rename_bmp_images(folder_path1, prefix=num, suffix='2')
rename_bmp_images(folder_path2, prefix=num, suffix='3')