import cv2
import numpy as np
import os
import argparse
# from svm import predict_image_array, load_model



#提取西红柿，使用S+L的图像
def extract_s_l(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    s_channel = hsv[:,:,1]
    l_channel = lab[:,:,0]
    result = cv2.add(s_channel, l_channel)
    return result

def find_reflection(image_path, threshold=190):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 应用阈值分割
    _, reflection = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return reflection

def otsu_threshold(image):

    # 将图像转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Otsu阈值分割
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

# 提取花萼，使用G-R的图像
def extract_g_r(image):
    # image = cv2.imread(image_path)
    g_channel = image[:,:,1]
    r_channel = image[:,:,2]
    result = cv2.subtract(cv2.multiply(g_channel, 1.5), r_channel)
    return result


#提取西红柿，使用R-B的图像
def extract_r_b(image_path):
    image = cv2.imread(image_path)
    r_channel = image[:,:,2]
    b_channel = image[:,:,0]
    result = cv2.subtract(r_channel, b_channel)
    return result

def extract_r_g(image_path):
    image = cv2.imread(image_path)
    r_channel = image[:,:,2]
    g_channel = image[:,:,1]
    result = cv2.subtract(r_channel, g_channel)
    return result

def threshold_segmentation(image, threshold, color=255):
    _, result = cv2.threshold(image, threshold, color, cv2.THRESH_BINARY)
    return result

def bitwise_operation(image1, image2, operation='and'):
    if operation == 'and':
        result = cv2.bitwise_and(image1, image2)
    elif operation == 'or':
        result = cv2.bitwise_or(image1, image2)
    else:
        raise ValueError("operation must be 'and' or 'or'")
    return result

def largest_connected_component(bin_img):
    # 使用connectedComponentsWithStats函数找到连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    # 找到最大的连通区域（除了背景）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 创建一个新的二值图像，只显示最大的连通区域
    new_bin_img = np.zeros_like(bin_img)
    new_bin_img[labels == largest_label] = 255

    return new_bin_img

def close_operation(bin_img, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    return closed_img

def open_operation(bin_img, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return opened_img


def draw_tomato_edge(original_img, bin_img):
    bin_img_processed = close_operation(bin_img, kernel_size=(15, 15))
    # cv2.imshow('Close Operation', bin_img_processed)
    # bin_img_processed = open_operation(bin_img_processed, kernel_size=(19, 19))
    # cv2.imshow('Open Operation', bin_img_processed)
    # 现在使用处理后的bin_img_processed查找轮廓
    contours, _ = cv2.findContours(bin_img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，直接返回原图
    if not contours:
        return original_img
    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    # 多边形近似的精度调整
    epsilon = 0.0006 * cv2.arcLength(max_contour, True)  # 可以调整这个值
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # 绘制轮廓
    cv2.drawContours(original_img, [approx], -1, (0, 255, 0), 3)
    mask = np.zeros_like(bin_img)

    # 使用白色填充最大轮廓
    cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)

    return original_img, mask

def draw_tomato_edge_convex_hull(original_img, bin_img):
    bin_img_blurred = cv2.GaussianBlur(bin_img, (5, 5), 0)
    contours, _ = cv2.findContours(bin_img_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_img
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)
    cv2.drawContours(original_img, [hull], -1, (0, 255, 0), 3)
    return original_img

# 得到完整的西红柿二值图像，除了绿色花萼
def fill_holes(bin_img):
    # 复制 bin_img 到 img_filled
    img_filled = bin_img.copy()

    # 获取图像的高度和宽度
    height, width = bin_img.shape

    # 创建一个掩码，比输入图像大两个像素点
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # 使用 floodFill 函数填充黑色区域
    cv2.floodFill(img_filled, mask, (0, 0), 255)

    # 反转填充后的图像
    img_filled_d = cv2.bitwise_not(img_filled)

    # 使用 bitwise_or 操作合并原图像和填充后的图像
    img_filled = cv2.bitwise_or(bin_img, img_filled)
    # 裁剪 img_filled 和 img_filled_d 到与 bin_img 相同的大小
    # img_filled = img_filled[:height, :width]
    img_filled_d = img_filled_d[:height, :width]

    return img_filled, img_filled_d

def bitwise_and_rgb_with_binary(rgb_img, bin_img):
    # 将二值图像转换为三通道图像
    bin_img_3channel = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # 使用 bitwise_and 操作合并 RGB 图像和二值图像
    result = cv2.bitwise_and(rgb_img, bin_img_3channel)

    return result


def extract_max_connected_area(image_path, lower_hsv, upper_hsv):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR转换到HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 使用阈值获取指定区域的二值图像
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 找到二值图像的连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 找到最大的连通区域（除了背景）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 创建一个新的二值图像，只显示最大的连通区域
    new_bin_img = np.zeros_like(mask)
    new_bin_img[labels == largest_label] = 255

    # 复制 new_bin_img 到 img_filled
    img_filled = new_bin_img.copy()

    # 获取图像的高度和宽度
    height, width = new_bin_img.shape

    # 创建一个掩码，比输入图像大两个像素点
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # 使用 floodFill 函数填充黑色区域
    cv2.floodFill(img_filled, mask, (0, 0), 255)

    # 反转填充后的图像
    img_filled_inv = cv2.bitwise_not(img_filled)

    # 使用 bitwise_or 操作合并原图像和填充后的图像
    img_filled = cv2.bitwise_or(new_bin_img, img_filled_inv)

    return img_filled




def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir_path', type=str, default=r'D:\project\Tomato\20240410RGBtest2\data',
                        help='the directory path of images')
    parser.add_argument('--threshold_s_l', type=int, default=180,
                        help='the threshold for s_l')
    parser.add_argument('--threshold_r_b', type=int, default=15,
                        help='the threshold for r_b')

    args = parser.parse_args()

    for img_file in os.listdir(args.dir_path):
        if img_file.endswith('.bmp'):
            img_path = os.path.join(args.dir_path, img_file)
            s_l = extract_s_l(img_path)
            otsu_thresholded = otsu_threshold(s_l)
            img_fore = bitwise_and_rgb_with_binary(cv2.imread(img_path), otsu_thresholded)
            img_fore_defect = extract_g_r(img_fore)
            img_fore_defect = threshold_segmentation(img_fore_defect, args.threshold_r_b)
            # cv2.imshow('img_fore_defect', img_fore_defect)
            thresholded_s_l = threshold_segmentation(s_l, args.threshold_s_l)
            new_bin_img = largest_connected_component(thresholded_s_l)
            zhongggggg = cv2.bitwise_or(new_bin_img, cv2.imread('defect_mask.bmp', cv2.IMREAD_GRAYSCALE))
            cv2.imshow('zhongggggg', zhongggggg)
            new_otsu_bin_img = largest_connected_component(otsu_thresholded)
            filled_img, defect = fill_holes(new_bin_img)
            defect = bitwise_and_rgb_with_binary(cv2.imread(img_path), defect)
            cv2.imshow('defect', defect)
            edge, mask = draw_tomato_edge(cv2.imread(img_path), new_bin_img)
            org_defect = bitwise_and_rgb_with_binary(edge, new_bin_img)
            fore = bitwise_and_rgb_with_binary(cv2.imread(img_path), mask)
            fore_g_r_t = threshold_segmentation(extract_g_r(fore), 20)
            fore_g_r_t_ture = bitwise_and_rgb_with_binary(cv2.imread(img_path), fore_g_r_t)
            cv2.imwrite('defect_big.bmp', fore_g_r_t_ture)
            res = cv2.bitwise_or(new_bin_img, fore_g_r_t)
            white = find_reflection(img_path)

            # SVM预测
            # 加载模型
            # model, scaler = load_model('/Users/xs/PycharmProjects/super-tomato/svm_green.joblib')

            # 对图像进行预测
            # predicted_mask = predict_image_array(image, model, scaler)



            cv2.imshow('white', white)

            cv2.imshow('fore', fore)
            cv2.imshow('fore_g_r_t', fore_g_r_t)
            cv2.imshow('mask', mask)
            print('mask', mask.shape)
            print('filled', filled_img.shape)
            print('largest', new_bin_img.shape)
            print('rp', org_defect.shape)
            cv2.imshow('res', res)

            # lower_hsv = np.array([19, 108, 15])
            # upper_hsv = np.array([118, 198, 134])
            # max_connected_area = extract_max_connected_area(img_path, lower_hsv, upper_hsv)
            # cv2.imshow('Max Connected Area', max_connected_area)

            # 显示原始图像
            original_img = cv2.imread(img_path)
            cv2.imshow('Original', original_img)
            cv2.imshow('thresholded_s_l', thresholded_s_l)
            cv2.imshow('Largest Connected Component', new_bin_img)
            cv2.imshow('Filled', filled_img)
            cv2.imshow('Defect', defect)
            cv2.imshow('Org_defect', org_defect)
            cv2.imshow('otsu_thresholded', new_otsu_bin_img)


            #显示轮廓
            cv2.imshow('Edge', edge)

            # 等待用户按下任意键
            cv2.waitKey(0)

            # 关闭所有窗口
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()