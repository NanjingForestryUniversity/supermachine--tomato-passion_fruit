import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from spec_read import all_spectral_data

def prepare_data(data):
    """Calculate the average spectral values for each fruit across all pixels."""
    return np.mean(data, axis=(1, 2))

def train_model(X, y):
    """Train a RandomForest model."""
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    return rf

def split_data(X, y, test_size=0.20, random_state=4):
    """Split data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return MSE and predictions."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred

def print_predictions(y_test, y_pred):
    """Print actual and predicted values."""
    print("Test Set Predictions:")
    for i, (real, pred) in enumerate(zip(y_test, y_pred)):
        print(f"Sample {i + 1}: True Value = {real:.2f}, Predicted Value = {pred:.2f}")

def plot_spectra(X, y):
    """Plot the average spectra for all samples and annotate with sweetness_acidity values."""
    plt.figure(figsize=(10, 6))
    for i in range(X.shape[0]):
        plt.plot(X[i], label=f'Sample {i+1}')
        plt.annotate(f'{y[i]:.1f}', xy=(len(X[i])-1, X[i][-1]), xytext=(5, 0),
                     textcoords='offset points', ha='left', va='center')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Average Spectral Value')
    plt.title('Average Spectral Curves for All Samples')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.show()

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
    plot_spectra(X, sweetness_acidity)  # 绘制光谱曲线并添加标注
    X_train, X_test, y_train, y_test = split_data(X, sweetness_acidity)
    rf_model = train_model(X_train, y_train)
    mse, y_pred = evaluate_model(rf_model, X_test, y_test)

    print("Transformed data shape:", X_train.shape)
    print("Mean Squared Error on the test set:", mse)
    print_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()