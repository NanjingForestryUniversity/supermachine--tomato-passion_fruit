import os
from PIL import Image

def convert_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    img.save(filepath)

# 调用函数，替换为你的目录路径
convert_images(r'D:\同步盘\project\Tomato\20240410RGBtest2\super-tomato\images')