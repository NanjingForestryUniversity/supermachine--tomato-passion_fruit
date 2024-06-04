import cv2
import numpy as np
import matplotlib.pyplot as plt

def dual_threshold_and_max_component(image_path, hue_value=37, hue_delta=10, value_target=30, value_delta=10):
    # 读取图像
    image = cv2.imread(image_path)

    # 检查图像是否读取成功
    if image is None:
        print("Error: Image could not be read.")
        return

    # 将图像从BGR转换到HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建H通道阈值掩码
    lower_hue = np.array([hue_value - hue_delta, 0, 0])
    upper_hue = np.array([hue_value + hue_delta, 255, 255])
    hue_mask = cv2.inRange(hsv_image, lower_hue, upper_hue)

    # 创建V通道排除中心值的掩码
    lower_value_1 = np.array([0, 0, 0])
    upper_value_1 = np.array([180, 255, value_target - value_delta])
    lower_value_2 = np.array([0, 0, value_target + value_delta])
    upper_value_2 = np.array([180, 255, 255])

    value_mask_1 = cv2.inRange(hsv_image, lower_value_1, upper_value_1)
    value_mask_2 = cv2.inRange(hsv_image, lower_value_2, upper_value_2)
    value_mask = cv2.bitwise_or(value_mask_1, value_mask_2)

    # 合并H通道和V通道掩码
    combined_mask = cv2.bitwise_and(hue_mask, value_mask)

    # 形态学操作 - 开运算，去除小的粘连
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, 4, cv2.CV_32S)

    # 找出最大的连通区域（除了背景）
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 跳过背景
    max_mask = (labels == max_label).astype(np.uint8) * 255

    # 使用掩码生成结果图像
    result_image = cv2.bitwise_and(image, image, mask=max_mask)
    # 设置背景为白色
    result_image[max_mask == 0] = [255, 255, 255]

    # 将结果图像从BGR转换到RGB以便正确显示
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # 使用matplotlib显示原始图像和结果图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image_rgb)
    plt.title('Largest Connected Component on White Background')
    plt.axis('off')

    plt.show()

# 使用函数
image_path = r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\data\passion_fruit_img\50.bmp'  # 替换为你的图片路径
dual_threshold_and_max_component(image_path)