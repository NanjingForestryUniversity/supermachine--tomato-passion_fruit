import logging
import sys
from typing import Optional

import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import ndimage
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import binom
import matplotlib.pyplot as plt
import time
import pickle
import os
import utils
from root_dir import ROOT_DIR


class Astragalin(object):
    def __init__(self, load_from=None, debug_mode=False, class_weight=None):
        if load_from is None:
            self.model = DecisionTreeClassifier(random_state=65, class_weight=class_weight)
        else:
            self.load(load_from)
        self.log = utils.Logger(is_to_file=debug_mode)
        self.debug_mode = debug_mode

    def load(self, path=None):
        if path is None:
            path = os.path.join(ROOT_DIR, 'models')
            model_files = os.listdir(path)
            if len(model_files) == 0:
                self.log.log("No model found!")
                return 1
            self.log.log("./ Models Found:")
            _ = [self.log.log("├--" + str(model_file)) for model_file in model_files]
            file_times = [model_file[6:-2] for model_file in model_files]
            latest_model = model_files[int(np.argmax(file_times))]
            self.log.log("└--Using the latest model: " + str(latest_model))
            path = os.path.join(ROOT_DIR, "models", str(latest_model))
        if not os.path.isabs(path):
            logging.warning('给的是相对路径')
            return -1
        if not os.path.exists(path):
            logging.warning('文件不存在')
            return -1
        with open(path, 'rb') as f:
            model_dic = pickle.load(f)
        self.model = model_dic['model']
        return 0

    def fit(self, data_x, data_y):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=65)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        print(confusion_matrix(y_test, y_pred))

        pre_score = accuracy_score(y_test, y_pred)
        self.log.log("Test accuracy is:" + str(pre_score * 100) + "%.")
        y_pred = self.model.predict(x_train)

        pre_score = accuracy_score(y_train, y_pred)
        self.log.log("Train accuracy is:" + str(pre_score * 100) + "%.")
        y_pred = self.model.predict(data_x)

        pre_score = accuracy_score(data_y, y_pred)
        self.log.log("Total accuracy is:" + str(pre_score * 100) + "%.")

        return int(pre_score * 100)

    def fit_value(self, file_name=None, data_path='data/1.txt', select_bands=[91, 92, 93, 94, 95, 96, 97, 98, 99, 100]):
        data_x, data_y = self.data_construction(data_path, select_bands)
        score = self.fit(data_x, data_y)
        print('score:', score)
        model_name = self.save(file_name=file_name)
        return score, model_name

    def save(self, file_name):
        # 保存模型
        if file_name is None:
            file_name = "model_" + time.strftime("%Y-%m-%d_%H-%M") + ".p"
        file_name = os.path.join(ROOT_DIR, "models", file_name)
        model_dic = {'model': self.model}
        with open(file_name, 'wb') as f:
            pickle.dump(model_dic, f)
        self.log.log("Model saved to '" + str(file_name) + "'.")
        return file_name

    # def data_construction(self, data_path, select_bands):
    #     data = utils.read_envi_ascii(data_path)
    #     beijing = data['beijing'][:, select_bands]
    #     zazhi1 = data['zazhi1'][:, select_bands]
    #     # zazhi2 = data['zazhi2'][:, select_bands]
    #     huangqi = data['huangqi'][:, select_bands]
    #     gancaopian = data['gancaopian'][:, select_bands]
    #     # hongqi = data['hongqi'][:, select_bands]
    #     beijing_y = np.zeros(beijing.shape[0])
    #     zazhi1_y = np.ones(zazhi1.shape[0]) * 3
    #     # zazhi2_y = np.ones(zazhi2.shape[0]) * 2
    #     huangqi_y = np.ones(huangqi.shape[0]) * 1
    #     gancaopian_y = np.ones(gancaopian.shape[0]) * 4
    #     # hongqi_y = np.ones(hongqi.shape[0]) * 5
    #     data_x = np.concatenate((beijing, zazhi1, huangqi, gancaopian), axis=0)
    #     data_y = np.concatenate((beijing_y, zazhi1_y, huangqi_y, gancaopian_y), axis=0)
    #     return data_x, data_y

    def data_construction(self, data_path='data/1.txt', select_bands=[91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
                          type=['beijing', 'zazhi1', 'huangqi', 'gancaopian']):
        '''
        :param data_path: 数据文件路径
        :param select_bands: 选择的波段
        :param type: 选择的类型
        :return: data_x, data_y
        '''
        data = utils.read_envi_ascii(data_path)
        # 判断读取的txt文件内是否有beijing和haungqi类型的数据
        if 'beijing' not in data or 'huangqi' not in data:
            logging.error("数据文件中缺少'beijing'或'huangqi'类型标签")
            raise ValueError("数据文件中缺少'beijing'或'huangqi'类型标签")
        data_x = np.concatenate([data[key][:, select_bands] for key in type], axis=0)
        data_y = np.concatenate([np.zeros(data[key].shape[0]) if key == 'beijing' else np.ones(data[key].shape[0])
            if key == 'huangqi' else np.ones(data[key].shape[0]) * (i + 2) for i, key in enumerate(type)], axis=0)
        return data_x, data_y

    def predict(self, data_x):
        '''
        对数据进行预测
        :param data_x: 波段选择后的数据
        :return: 预测结果二值化后的数据，0为背景，1为黄芪,2为杂质2，3为杂质1，4为甘草片，5为红芪
        '''
        data_x_shape = data_x.shape
        data_x = data_x.reshape(-1, data_x.shape[2])
        data_y = self.model.predict(data_x)
        data_y = data_y.reshape(data_x_shape[0], data_x_shape[1]).astype(np.uint8)
        data_y, centers, categories = self.connect_space(data_y)
        result = {'data_y': data_y, 'centers': centers, 'categories': categories}
        return result

    def connect_space(self, data_y):
        # 连通域处理离散点
        labels, num_features = ndimage.label(data_y)
        centers = []
        categories = []
        for i in range(1, num_features + 1):
            mask = (labels == i)
            counts = np.bincount(data_y[mask])
            category = np.argmax(counts)
            data_y[mask] = category
            center = ndimage.measurements.center_of_mass(data_y, labels, [i])
            center = list(center)
            center = np.array(center).astype(int)
            centers.append(center)
            categories.append(category)
        return data_y, centers, categories


if __name__ == '__main__':
    detector = Astragalin()
    detector.fit_value(file_name="astragalin.p", data_path="data/1.txt")