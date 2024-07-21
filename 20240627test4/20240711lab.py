import cv2
import numpy as np
import os

# 读取文件夹中的所有图片文件
def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

# Lab颜色空间的a阈值分割，同时处理灰度值大于190的像素
def threshold_lab_a_and_high_gray(image, lower_threshold=0, upper_threshold=20):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    _, a, _ = cv2.split(lab_image)

    # 创建一个与a通道相同大小的黑色图像
    binary_image = np.zeros_like(a)

    # 将a通道中值在指定范围内的像素设置为白色（255）
    binary_image[(a >= lower_threshold) & (a <= upper_threshold)] = 255

    # 为灰度值大于190的像素创建二值图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_gray_image = np.zeros_like(gray_image)
    high_gray_image[gray_image > 170] = 255

    # 从a通道阈值图中移除灰度值大于190的像素
    final_image = cv2.bitwise_and(binary_image, binary_image, mask=np.bitwise_not(high_gray_image))

    return binary_image, high_gray_image, final_image

# 拼接并显示所有图片
def concatenate_images(original, images, filename, scale=0.5):
    # 将所有单通道图像转换为三通道图像
    resized_imgs = []
    for img in images:
        if len(img.shape) == 2:  # 单通道图像
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 缩放图像
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        resized_imgs.append(img)

    # 将原图也转换为相同大小和缩放
    resized_original = cv2.resize(original, (int(original.shape[1] * scale), int(original.shape[0] * scale)))

    # 水平拼接第一行和第二行
    top_row = cv2.hconcat([resized_original, resized_imgs[0]])
    bottom_row = cv2.hconcat([resized_imgs[1], resized_imgs[2]])

    # 垂直拼接所有行
    final_image = cv2.vconcat([top_row, bottom_row])

    # 显示图片
    cv2.imshow(f"Combined Images - {filename}", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    folder = r'F:\images'  # 替换为你的文件夹路径
    images = read_images_from_folder(folder)

    for filename, image in images:
        lab_thresh, high_gray, final_image = threshold_lab_a_and_high_gray(image, lower_threshold=115, upper_threshold=135)
        concatenate_images(image, [lab_thresh, high_gray, final_image], filename, scale=0.5)  # 添加缩放因子

if __name__ == "__main__":
    main()

