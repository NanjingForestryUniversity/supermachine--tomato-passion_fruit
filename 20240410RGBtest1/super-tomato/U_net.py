
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# 定义 U-Net 的下采样（编码器）部分
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

# 定义 U-Net 的上采样（解码器）部分
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算 padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 组装完整的 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UNetDown(3, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = UNetUp(1024, 512)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(128, 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x_middle = self.middle_conv(x4)
        x = self.up1(x_middle, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_conv(x)
        return torch.sigmoid(x)

# 创建模型
model = UNet()
print(model)


class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, size=(320, 458)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.size = size
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.bmp', '.png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 确保是单通道的灰度图

        # Resize image and mask
        resize = transforms.Resize(self.size)
        image = resize(image)
        mask = resize(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 128).type(torch.FloatTensor)  # 二值化标签
        return image, mask

transform = transforms.Compose([
    transforms.ToTensor(),
])


def test_and_save(model, test_dir, save_dir):
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_images = os.listdir(test_dir)
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # 添加 batch dimension

        with torch.no_grad():
            prediction = model(image)

        prediction = prediction.squeeze(0).squeeze(0)  # 去掉 batch 和 channel dimension
        prediction = (prediction > 0.5).type(torch.uint8)  # 二值化
        save_image = Image.fromarray(prediction.numpy() * 255, 'L')
        save_image.save(os.path.join(save_dir, img_name))


def train(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for images, masks in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

train_dataset = ImageDataset('/Users/xs/PycharmProjects/super-tomato/datasets_green/train/img', '/Users/xs/PycharmProjects/super-tomato/datasets_green/train/label', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 上面定义的 UNet 类的代码
model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型


train(model, train_loader, optimizer, criterion, epochs=50)



test_and_save(model, '/Users/xs/PycharmProjects/super-tomato/tomato_img_25', '/Users/xs/PycharmProjects/super-tomato/tomato_img_25/pre')
