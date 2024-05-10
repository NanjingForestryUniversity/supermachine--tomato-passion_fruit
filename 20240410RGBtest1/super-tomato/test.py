import cv2
import numpy as np


def find_reflection(image_path, threshold=190):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 应用阈值分割
    _, reflection = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return reflection

def repair_reflection_telea(image_path, reflection, inpaint_radius=20):
    # 读取图像
    image = cv2.imread(image_path)

    # 将高亮区域转换为二值图像
    _, reflection_binary = cv2.threshold(reflection, 1, 255, cv2.THRESH_BINARY)

    # 使用inpaint函数修复高亮区域
    repaired_image = cv2.inpaint(image, reflection_binary, inpaint_radius, cv2.INPAINT_TELEA)

    return repaired_image

# 读取图像
image_path = '/Users/xs/PycharmProjects/super-tomato/tomato_img_25/60.bmp'  # 替换为你的图像路径
image = find_reflection(image_path)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 修复反光
image = repair_reflection_telea(image_path, image)
cv2.imshow('ima11ge', cv2.imread(image_path))
# 创建窗口
cv2.namedWindow('image')

# 创建滑动条
cv2.createTrackbar('Threshold', 'image', 0, 255, lambda x: None)

while True:
    # 获取滑动条的值
    threshold = cv2.getTrackbarPos('Threshold', 'image')

    # 使用阈值进行分割
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 显示二值图像
    cv2.imshow('image', thresholded_image)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 销毁所有窗口
cv2.destroyAllWindows()