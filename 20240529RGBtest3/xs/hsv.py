import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_mask(hsv_image, hue_value, hue_delta, value_target, value_delta):
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
    value_mask_1 = cv2.bitwise_not(value_mask_1)
    cv2.imshow('value_mask_1', value_mask_1)
    value_mask_2 = cv2.inRange(hsv_image, lower_value_2, upper_value_2)
    cv2.imshow('value_mask_2', value_mask_2)
    value_mask = cv2.bitwise_and(value_mask_1, value_mask_2)
    cv2.imshow('value_mask', value_mask)
    # 等待用户按下任意键
    cv2.waitKey(0)

    # 关闭所有窗口
    cv2.destroyAllWindows()

    # 合并H通道和V通道掩码
    return cv2.bitwise_and(hue_mask, value_mask)

def apply_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def find_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    if num_labels < 2:
        return None  # No significant components found
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background
    return (labels == max_label).astype(np.uint8) * 255

def process_image(image_path, hue_value=37, hue_delta=10, value_target=25, value_delta=10):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} could not be read.")
        return None

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined_mask = create_mask(hsv_image, hue_value, hue_delta, value_target, value_delta)
    combined_mask = apply_morphology(combined_mask)
    max_mask = find_largest_component(combined_mask)
    cv2.imshow('max_mask', max_mask)
    # 等待用户按下任意键
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()

    if max_mask is None:
        print(f"No significant components found in {image_path}.")
        return None

    result_image = cv2.bitwise_and(image, image, mask=max_mask)
    result_image[max_mask == 0] = [255, 255, 255]  # Set background to white
    return result_image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".bmp"):
            image_path = os.path.join(input_folder, filename)
            result_image = process_image(image_path)
            if result_image is not None:
                output_path = os.path.join(output_folder, filename)
                save_image(result_image, output_path)
                print(f"Processed and saved {filename} to {output_folder}.")


# 主函数调用
input_folder = r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\data\passion_fruit_img'  # 替换为你的输入文件夹路径
output_folder = r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\data\01'  # 替换为你的输出文件夹路径
process_images_in_folder(input_folder, output_folder)