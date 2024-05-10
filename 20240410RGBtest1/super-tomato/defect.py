import cv2
import numpy as np


def segment_image_by_variance(image_path, m, n, variance_threshold):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return None

    # 图像的高度和宽度
    h, w = image.shape

    # 计算每个块的尺寸
    block_h, block_w = h // m, w // n

    # 创建空白图像
    segmented_image = np.zeros_like(image)

    # 遍历每个块
    for row in range(m):
        for col in range(n):
            # 计算块的位置
            y1, x1 = row * block_h, col * block_w
            y2, x2 = y1 + block_h, x1 + block_w

            # 提取块
            block = image[y1:y2, x1:x2]

            # 计算方差
            variance = np.var(block)

            # 根据方差设置新图像的对应区块
            if variance > variance_threshold:
                segmented_image[y1:y2, x1:x2] = 1
            else:
                segmented_image[y1:y2, x1:x2] = 0

    # 将新图像的值扩展到0-255范围，以便可视化
    segmented_image *= 255

    return segmented_image


# 示例用法
image_path = '/Users/xs/PycharmProjects/super-tomato/tomato_img_25/60.bmp'  # 替换为你的番茄图像路径
m, n = 300, 300  # 划分的区块数
variance_threshold = 80  # 方差的阈值

segmented_image = segment_image_by_variance(image_path, m, n, variance_threshold)

if segmented_image is not None:
    cv2.imshow("Segmented Image", segmented_image)
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
