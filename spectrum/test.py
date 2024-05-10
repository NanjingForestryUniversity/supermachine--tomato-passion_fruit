import numpy as np
import cv2
import copy
import pandas as pd
import os

# 指定文件夹路径
folder_path = 'data'  # 替换成你的文件夹路径
mask_folder_path = 'result'  # 替换成你的掩码图像文件夹路径

# 获取文件夹中的所有文件
files = os.listdir(folder_path)
files1 = os.listdir(mask_folder_path)

D =[]
for i in range(39, 40):
    with open(folder_path + '/' + str(i) + '_ref.hdr', 'r') as hdr_file:
        lines = hdr_file.readlines()
        for line in lines:
            if line.startswith('lines'):
                height = int(line.split()[-1])
            elif line.startswith('samples'):
                width = int(line.split()[-1])
            elif line.startswith('bands'):
                bands = int(line.split()[-1])
    raw_image = np.fromfile(folder_path + '/' + str(i) + '_ref.raw', dtype='float32')
    mask = cv2.imread(mask_folder_path + '/' + str(i) + '.tiff', cv2.IMREAD_GRAYSCALE)
    formatImage = np.zeros((height, width, bands))
    for row in range(0, height):
        for dim in range(0, bands):
            formatImage[row,:,dim] = raw_image[(dim + row*bands) * width:(dim + 1 + row*bands)* width]
    # 创建一个和光谱图像形状相同的全零数组，用于存储西红柿区域的光谱图像
    tomato_spectrum_image = np.zeros_like(formatImage)
    # 将掩码图像扩展到和光谱图像形状相同，以便进行按位与操作
    mask = np.stack([mask]*bands, axis=2)
    # 使用掩码提取西红柿区域的光谱图像
    tomato_spectrum_image = np.where(mask == 255, formatImage, 0)
    # print(tomato_spectrum_image.shape)
    # cv2.imshow('', tomato_spectrum_image[:, :, 145])
    # cv2.waitKey(0)
    data_save = []
    for i in range(0, 224):
        data = copy.deepcopy(tomato_spectrum_image[:, :, i])
        data[data>0] = 1
        num = np.sum(data)
        print(num)
        average_values_tomato = np.sum(tomato_spectrum_image[:, :, i])/num
        print("西红柿区域各波段的平均值:", average_values_tomato)
        data_save.append(average_values_tomato)
    print(data_save)
    D.append(data_save)
# 创建一个DataFrame来存储结果
D_array = np.array(D).reshape(224, 1)
df = pd.DataFrame(D_array)

# 将DataFrame导出为Excel文件
df.to_excel('test39.xlsx', index=False)



#选择通道数为12/46/96的三个通道相当于rgb三通道的图片
# imgR = formatImage[:,:,12]
# imgG = formatImage[:,:,46]
# imgB = formatImage[:,:,96]
#
# rgbImg = cv2.merge([imgR, imgG, imgB])
# cv2.imshow('test', rgbImg)
# cv2.waitKey()

#
# # 使用图像的宽度、高度和波段数来重塑raw光谱图像的形状
# height = 756 # 替换成你的图像高度
# width = 1200  # 替换成你的图像宽度
# bands = 224  # 替换成你的图像波段数
# spectrum_image = raw_image.reshape((height, width, bands))
# # 读取mask掩码图像，假设掩码图像是一个二值图像，西红柿的区域为255，其他区域为0
# mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
# # 创建一个和光谱图像形状相同的全零数组，用于存储西红柿区域的光谱图像
# tomato_spectrum_image = np.zeros_like(spectrum_image)
#
# # 将掩码图像扩展到和光谱图像形状相同，以便进行按位与操作
# mask = np.stack([mask]*bands, axis=2)
#
# # 使用掩码提取西红柿区域的光谱图像
# tomato_spectrum_image = np.where(mask == 255, spectrum_image, 0)
# print(tomato_spectrum_image.shape)
# # data_save = []
# # for i in range(0, 224):
# #     data = copy.deepcopy(tomato_spectrum_image[:, :, i])
# #     data[data>0] = 1
# #     num = np.sum(data)
# #     print(num)
# #     average_values_tomato = np.sum(tomato_spectrum_image[:, :, i])/num
# #     print("西红柿区域各波段的平均值:", average_values_tomato)
# #     data_save.append(average_values_tomato)
# # print(data_save)
# # 计算每个谱段上西红柿的光谱信息均值
# # average_values_tomato = np.mean(tomato_spectrum_image, axis=(0, 1))
#
# # print(tomato_spectrum_image[:, :, 145])
#
# # cv2.imshow('', tomato_spectrum_image[:, :, 145])
# # cv2.waitKey(0)
# # 打印结果
# # print("西红柿区域各波段的平均值:", average_values_tomato)
# # import pandas as pd
# #
# # # 创建一个DataFrame来存储结果
# # df = pd.DataFrame(average_values_tomato, columns=['Average Spectral Values'])

# # 将DataFrame导出为Excel文件
# df.to_excel('39.xlsx', index=False)