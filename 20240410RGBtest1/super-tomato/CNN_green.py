import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class ImageDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert('RGB')).transpose((2, 0, 1))  # 转换为RGB图像，确保有3个通道
        if self.mask_paths[0] is not None:
            mask = np.array(Image.open(self.mask_paths[idx]).convert('I'))  # 转换为32位深度的灰度图像
            mask = mask / np.max(mask)  # Normalize to 0-1
            return img, mask[np.newaxis, :]
        else:
            return img, None

def train_model(dataloader, model, criterion, optimizer, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_accuracy = 0.0
    for epoch in tqdm(range(epochs), desc="Training"):
        for img, mask in dataloader:
            img = img.float().to(device)
            mask = mask.float().to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            # 二值化模型的输出
            preds = outputs.detach().cpu().numpy() > 0.5
            mask = (mask.cpu().numpy() > 0.5)  # Binarize the mask

            # 计算准确度、精度和召回率
            accuracy = accuracy_score(mask.flatten(), preds.flatten())
            precision = precision_score(mask.flatten(), preds.flatten())
            recall = recall_score(mask.flatten(), preds.flatten())

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

            # 如果这个模型的准确度更好，就保存它
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model.pth')

    return model

def predict(model, img_path):
    img = np.array(Image.open(img_path)).transpose((2, 0, 1))  # 调整维度为(C, H, W)
    img = torch.from_numpy(img).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
    pred = outputs.squeeze().numpy()
    return pred

def main(train_img_folder, train_mask_folder, test_img_folder, test_mask_folder, epochs, img_path='/Users/xs/PycharmProjects/super-tomato/datasets_green/test/label'):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = SimpleCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Create data loaders
    train_dataset = ImageDataset(train_img_folder, train_mask_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    # Train model
    model = train_model(train_dataloader, model, criterion, optimizer, epochs)

    # Create test data loaders
    test_dataset = ImageDataset(test_img_folder, test_mask_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Use trained model to predict
    for img, mask in test_dataloader:
        img = img.float().to(device)
        mask = mask.float().to(device)

        start_time = time.time()
        outputs = model(img)
        elapsed_time = time.time() - start_time

        # Binarize model's output
        preds = outputs.detach().cpu().numpy() > 0.5
        mask = mask.cpu().numpy()

        # Calculate accuracy, precision and recall
        accuracy = accuracy_score(mask.flatten(), preds.flatten())
        precision = precision_score(mask.flatten(), preds.flatten())
        recall = recall_score(mask.flatten(), preds.flatten())

        print(f'Prediction for {img_path} saved, Time: {elapsed_time:.3f} seconds, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
# 调用函数示例
main('/Users/xs/PycharmProjects/super-tomato/datasets_green/train/img',
     '/Users/xs/PycharmProjects/super-tomato/datasets_green/train/label',
     '/Users/xs/PycharmProjects/super-tomato/datasets_green/test/img',
     '/Users/xs/PycharmProjects/super-tomato/datasets_green/test/label', 1)



def predict_and_display(model_path, img_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    dataset = ImageDataset(img_paths, [None]*len(img_paths))  # 我们不需要掩码，所以传入一个空列表
    dataloader = DataLoader(dataset, batch_size=1)

    for i, img in enumerate(dataloader):
        img = img.float().to(device)
        with torch.no_grad():
            outputs = model(img)
        pred = outputs.detach().cpu().numpy() > 0.5

        # 显示预测结果
        plt.imshow(pred[0, 0, :, :], cmap='gray')
        plt.title(f'Predicted Mask for {img_paths[i]}')
        plt.show()

# 调用函数示例
predict_and_display('best_model.pth', ['/Users/xs/PycharmProjects/super-tomato/datasets_green/test/img/5.bmp'])