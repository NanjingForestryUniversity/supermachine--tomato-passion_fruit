# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 15:40
# @Author  : TG
# @File    : 01.py
# @Software: PyCharm
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def prepare_data(data):
    """Reshape data and select specified spectral bands."""
    reshaped_data = data.reshape(data.shape[0], -1)  # 使用动态批量大小
    selected_bands = [8, 9, 10, 48, 49, 50, 77, 80, 103, 108, 115, 143, 145]
    return reshaped_data[:, selected_bands]

class SpectralModelingAndPrediction:
    def __init__(self, model_paths=None):
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
            "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
        }
        self.model_paths = model_paths or {}

    def split_data(self, X, y, test_size=0.20, random_state=12):
        """Split data into training and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return MSE and predictions."""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    def print_predictions(self, y_test, y_pred, model_name):
        """Print actual and predicted values."""
        print(f"Test Set Predictions for {model_name}:")
        for i, (real, pred) in enumerate(zip(y_test, y_pred)):
            print(f"Sample {i + 1}: True Value = {real:.2f}, Predicted Value = {pred:.2f}")

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            if model_name in self.model_paths:
                joblib.dump(model, self.model_paths[model_name])

            mse, y_pred = self.evaluate_model(model, X_test, y_test)
            print(f"Model: {model_name}")
            print(f"Mean Squared Error on the test set: {mse}")
            self.print_predictions(y_test, y_pred, model_name)
            print("\n" + "-" * 50 + "\n")

    def load_model(self, model_path):
        """加载模型"""
        return joblib.load(model_path)

    def read_spectral_data(self, hdr_path, raw_path):
        """读取光谱数据"""
        with open(hdr_path, 'r', encoding='latin1') as hdr_file:
            lines = hdr_file.readlines()
            height = width = bands = 0
            for line in lines:
                if line.startswith('lines'):
                    height = int(line.split()[-1])
                elif line.startswith('samples'):
                    width = int(line.split()[-1])
                elif line.startswith('bands'):
                    bands = int(line.split()[-1])

        raw_image = np.fromfile(raw_path, dtype='uint16')
        formatImage = np.zeros((height, width, bands))

        for row in range(height):
            for dim in range(bands):
                formatImage[row, :, dim] = raw_image[(dim + row * bands) * width:(dim + 1 + row * bands) * width]

        target_height, target_width, target_bands = 30, 30, 224
        formatImage = self._crop_or_pad(formatImage, height, width, bands, target_height, target_width, target_bands)
        return formatImage

    def _crop_or_pad(self, formatImage, height, width, bands, target_height, target_width, target_bands):
        """裁剪或填充图像"""
        if height > target_height:
            formatImage = formatImage[:target_height, :, :]
        elif height < target_height:
            pad_height = target_height - height
            formatImage = np.pad(formatImage, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

        if width > target_width:
            formatImage = formatImage[:, :target_width, :]
        elif width < target_width:
            pad_width = target_width - width
            formatImage = np.pad(formatImage, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        if bands > target_bands:
            formatImage = formatImage[:, :, :target_bands]
        elif bands < target_bands:
            pad_bands = target_bands - bands
            formatImage = np.pad(formatImage, ((0, 0), (0, 0), (0, pad_bands)), mode='constant', constant_values=0)

        return formatImage

    def predict(self, data, model_name):
        """预测数据"""
        model = self.load_model(self.model_paths[model_name])
        return model.predict(data)

    def run_training_and_prediction(self, training_data, training_target, prediction_directory):
        """运行训练和预测流程"""
        # 将数据重塑为2维
        training_data = training_data.reshape(training_data.shape[0], -1)

        # 训练阶段
        X_train, X_test, y_train, y_test = self.split_data(training_data, training_target)
        self.fit_and_evaluate(X_train, y_train, X_test, y_test)

        # 预测阶段
        all_spectral_data = []
        for i in range(1, 101):
            hdr_path = os.path.join(prediction_directory, f'{i}.HDR')
            raw_path = os.path.join(prediction_directory, f'{i}')
            if not os.path.exists(hdr_path) or not os.path.exists(raw_path):
                print(f"File {hdr_path} or {raw_path} does not exist.")
                continue
            spectral_data = self.read_spectral_data(hdr_path, raw_path)
            all_spectral_data.append(spectral_data)

        if not all_spectral_data:
            print("No spectral data was read. Please check the file paths and try again.")
            return

        all_spectral_data = np.stack(all_spectral_data)
        print(all_spectral_data.shape)  # This should print (100, 30, 30, 224) or fewer if some files are missing

        data_prepared = prepare_data(all_spectral_data)
        for model_name in self.models.keys():
            predictions = self.predict(data_prepared, model_name)
            print(f"Predictions for {model_name}:")
            print(predictions)
            print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    model_paths = {
        "RandomForest": '../20240529RGBtest3/models/random_forest_model_3.joblib',
        "GradientBoosting": '../20240529RGBtest3/models/gradient_boosting_model_3.joblib',
        "SVR": '../20240529RGBtest3/models/svr_model_3.joblib',
    }

    sweetness_acidity = np.array([
        16.2, 16.1, 17, 16.9, 16.8, 17.8, 18.1, 17.2, 17, 17.2, 17.1, 17.2,
        17.2, 17.2, 18.1, 17, 17.6, 17.4, 17.1, 17.1, 16.9, 17.6, 17.3, 16.3,
        16.5, 18.7, 17.6, 16.2, 16.8, 17.2, 16.8, 17.3, 16, 16.6, 16.7, 16.7,
        17.3, 16.3, 16.8, 17.4, 17.3, 16.3, 16.1, 17.2, 18.6, 16.8, 16.1, 17.2,
        18.3, 16.5, 16.6, 17, 17, 17.8, 16.4, 18, 17.7, 17, 18.3, 16.8, 17.5,
        17.7, 18.5, 18, 17.7, 17, 18.3, 18.1, 17.4, 17.7, 17.8, 16.3, 17.1, 16.8,
        17.2, 17.5, 16.6, 17.7, 17.1, 17.7, 19.4, 20.3, 17.3, 15.8, 18, 17.7,
        17.2, 15.2, 18, 18.4, 18.3, 15.7, 17.2, 18.6, 15.6, 17, 16.9, 17.4, 17.8,
        16.5
    ])

    # Specify the directory containing the HDR and RAW files
    directory = r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\xs\光谱数据3030'

    modeling = SpectralModelingAndPrediction(model_paths)

    # Initialize a list to hold all the spectral data arrays
    all_spectral_data = []

    # Loop through each data set (assuming there are 100 datasets)
    for i in range(1, 101):
        hdr_path = os.path.join(directory, f'{i}.HDR')
        raw_path = os.path.join(directory, f'{i}')

        # Check if files exist
        if not os.path.exists(hdr_path) or not os.path.exists(raw_path):
            print(f"File {hdr_path} or {raw_path} does not exist.")
            continue

        # Read data
        spectral_data = modeling.read_spectral_data(hdr_path, raw_path)
        all_spectral_data.append(spectral_data)

    # Stack all data into a single numpy array if not empty
    if all_spectral_data:
        all_spectral_data = np.stack(all_spectral_data)
        print(all_spectral_data.shape)  # This should print (100, 30, 30, 224) or fewer if some files are missing

        # Run training and prediction
        modeling.run_training_and_prediction(all_spectral_data, sweetness_acidity, directory)
    else:
        print("No spectral data was read. Please check the file paths and try again.")
