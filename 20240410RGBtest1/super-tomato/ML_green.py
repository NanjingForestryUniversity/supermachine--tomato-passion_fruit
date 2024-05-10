import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import os
import time


def load_image_data(img_path, label_path):
    """加载图像和标签，并将其转换为模型输入格式."""
    image = Image.open(img_path)
    label = Image.open(label_path).convert('L')

    image_np = np.array(image)
    label_np = np.array(label)

    # 转换图像数据格式
    n_samples = image_np.shape[0] * image_np.shape[1]
    n_features = image_np.shape[2]  # RGB通道
    image_np = image_np.reshape((n_samples, n_features))
    label_np = label_np.reshape((n_samples,))

    # 二值化标签
    label_np = (label_np > 128).astype(int)

    return image_np, label_np


def train_model(X, y, n_estimators=100):
    """训练模型."""
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return model


def predict_and_save(model, image_path, output_path):
    """预测并保存结果图像."""
    image = Image.open(image_path)
    image_np = np.array(image)
    n_samples = image_np.shape[0] * image_np.shape[1]
    image_np = image_np.reshape((n_samples, -1))

    # 预测
    predicted_labels = model.predict(image_np)
    predicted_labels = predicted_labels.reshape((image.size[1], image.size[0]))  # Use correct dimensions

    # 保存预测结果
    output_image = Image.fromarray((predicted_labels * 255).astype('uint8'), 'L')  # 'L' for grayscale
    output_image.save(output_path)



def process_folder(train_folder):
    X, y = [], []
    img_folder = os.path.join(train_folder, "img")
    label_folder = os.path.join(train_folder, "label")

    for filename in os.listdir(img_folder):
        img_path = os.path.join(img_folder, filename)
        label_path = os.path.join(label_folder, filename.replace('.bmp', '.png'))

        img_data, label_data = load_image_data(img_path, label_path)
        X.append(img_data)
        y.append(label_data)

    # 将数据列表转换为numpy数组
    X = np.vstack(X)
    y = np.concatenate(y)

    # 训练模型
    return train_model(X, y)


# 示例用法
train_folder = '/Users/xs/PycharmProjects/super-tomato/datasets_green/train'
t1 = time.time()
model = process_folder(train_folder)
t2 = time.time()
print(f'训练模型所需时间: {t2 - t1:.2f}秒')

# 测试图像处理和保存预测结果
test_folder = '/Users/xs/PycharmProjects/super-tomato/tomato_img_25'
for test_filename in os.listdir(test_folder):
    test_image_path = os.path.join(test_folder, test_filename)
    output_path = os.path.join(test_folder, "predicted_" + test_filename)
    predict_and_save(model, test_image_path, output_path)
