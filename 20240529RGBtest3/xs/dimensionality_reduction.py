import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from spec_read import all_spectral_data
import joblib

# def prepare_data(data):
#     """Reshape data and select specified spectral bands."""
#     reshaped_data = data.reshape(100, -1)
#     selected_bands = [8, 9, 10, 48, 49, 50, 77, 80, 103, 108, 115, 143, 145]
#     return reshaped_data[:, selected_bands]

def prepare_data(data):
    """Reshape data and select specified spectral bands."""
    selected_bands = [8, 9, 10, 48, 49, 50, 77, 80, 103, 108, 115, 143, 145]
    # 筛选特定的波段
    data_selected = data[:, :25, :, selected_bands]
    print(f'筛选后的数据尺寸：{data_selected.shape}')
    # 将筛选后的数据重塑为二维数组，每行代表一个样本
    reshaped_data = data_selected.reshape(-1, 25 * 30 * len(selected_bands))
    return reshaped_data


def split_data(X, y, test_size=0.20, random_state=12):
    """Split data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return multiple metrics and predictions."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred


def print_predictions(y_test, y_pred, model_name):
    """Print actual and predicted values."""
    print(f"Test Set Predictions for {model_name}:")
    for i, (real, pred) in enumerate(zip(y_test, y_pred)):
        print(f"Sample {i + 1}: True Value = {real:.2f}, Predicted Value = {pred:.2f}")

def main():
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

    X = prepare_data(all_spectral_data)
    print(f'原数据尺寸：{all_spectral_data.shape};训练数据尺寸：{X.shape}')
    X_train, X_test, y_train, y_test = split_data(X, sweetness_acidity)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        if model_name == "RandomForest":
            joblib.dump(model,
                        r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\models\passion_fruit.joblib')

        mse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)
        print(f"Model: {model_name}")
        print(f"MSE on the test set: {mse}")
        print(f"MAE on the test set: {mae}")
        print(f"R² score on the test set: {r2}")
        print_predictions(y_test, y_pred, model_name)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()