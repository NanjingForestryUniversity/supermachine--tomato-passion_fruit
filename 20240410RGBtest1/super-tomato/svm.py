import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import time
import os
import joblib


def load_model(model_path):
    # 加载模型和标准化器
    model, scaler = joblib.load(model_path)
    return model, scaler

def predict_image_array(image_array, model_path):
    # 加载模型和标准化器
    model, scaler = load_model(model_path)

    # 将图像转换为像素数组
    test_pixels = image_array.reshape(-1, 3)

    # 标准化
    test_pixels_scaled = scaler.transform(test_pixels)

    # 预测
    predictions = model.predict(test_pixels_scaled)

    # 转换预测结果为图像
    mask_predicted = predictions.reshape(image_array.shape[0], image_array.shape[1])

    return mask_predicted
def prepare_data(image_dir, mask_dir):
    # 初始化像素和标签列表
    all_pixels = []
    all_labels = []

    # 获取图像和掩码文件名列表
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # 遍历所有图像和掩码文件
    for image_file, mask_file in zip(image_files, mask_files):
        # 读取原始图像和掩码图像
        image = cv2.imread(os.path.join(image_dir, image_file))
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)

        # 提取像素
        pixels = image.reshape(-1, 3)  # 将图像转换为(n_pixels, 3)
        labels = (mask.reshape(-1) > 128).astype(int)  # 标记为0或1

        # 添加到列表
        all_pixels.append(pixels)
        all_labels.append(labels)

    # 将列表转换为NumPy数组
    all_pixels = np.concatenate(all_pixels, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_pixels, all_labels

# 加载数据
train_pixels, train_labels = prepare_data('/Users/xs/PycharmProjects/super-tomato/datasets_green/train-2/img',
                                          '/Users/xs/PycharmProjects/super-tomato/datasets_green/train-2/label')

# 数据标准化
scaler = StandardScaler()
train_pixels_scaled = scaler.fit_transform(train_pixels)

# 创建SVM模型
# model = svm.SVC(kernel='linear', C=1.0)
# model.fit(train_pixels_scaled, train_labels)
# # 在训练模型后保存模型
# joblib.dump((model, scaler), '/Users/xs/PycharmProjects/super-tomato/svm_green.joblib')  # 替换为你的模型文件路径

print('模型训练完成！')

def predict_image(image_path, model, scaler):
    # 读取图像
    image = cv2.imread(image_path)
    test_pixels = image.reshape(-1, 3)

    # 标准化
    test_pixels_scaled = scaler.transform(test_pixels)

    # 预测
    predictions = model.predict(test_pixels_scaled)

    # 转换预测结果为图像
    mask_predicted = predictions.reshape(image.shape[0], image.shape[1])

    return mask_predicted


# 对一个新的图像进行预测
time1 = time.time()
model, scaler = load_model('/Users/xs/PycharmProjects/super-tomato/svm_green.joblib')

predicted_mask = predict_image('/Users/xs/PycharmProjects/super-tomato/defect_big.bmp', model, scaler)
cv2.imwrite('/Users/xs/PycharmProjects/super-tomato/defect_mask.bmp', (predicted_mask * 255).astype('uint8'))
cv2.imshow('Predicted Mask', (predicted_mask * 255).astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

time2 = time.time()
print(f'预测时间: {time2 - time1:.2f}秒')
