import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from spec_read import all_spectral_data

def prepare_data(data):
    """Calculate the average spectral values and their gradients for each fruit across all pixels."""
    avg_spectra = np.mean(data, axis=(1, 2))
    gradients = np.gradient(avg_spectra, axis=1)
    second_gradients = np.gradient(gradients, axis=1)
    return avg_spectra, gradients, second_gradients

def train_model(X, y):
    """Train a RandomForest model."""
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    return rf

def split_data(X, y, test_size=0.20, random_state=2):
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

def plot_gradients(gradients):
    """Plot the gradient of the average spectra for all samples."""
    plt.figure(figsize=(10, 6))
    for i in range(gradients.shape[0]):
        plt.plot(gradients[i], label=f'Sample {i+1}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Gradient Value')
    plt.title('Gradient of Average Spectral Curves for All Samples')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.show()

def plot_second_gradients(second_gradients):
    """Plot the second gradient of the average spectra for all samples."""
    plt.figure(figsize=(10, 6))
    for i in range(second_gradients.shape[0]):
        plt.plot(second_gradients[i], label=f'Sample {i+1}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Second Gradient Value')
    plt.title('Second Gradient of Average Spectral Curves for All Samples')
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

    X_avg, X_grad, X_second_grad = prepare_data(all_spectral_data)

    plot_spectra(X_avg, sweetness_acidity)  # Plot average spectral curves
    plot_gradients(X_grad)  # Plot gradient curves
    plot_second_gradients(X_second_grad)  # Plot second gradient curves

    # Train and evaluate using average spectral values
    X_train_avg, X_test_avg, y_train_avg, y_test_avg = split_data(X_avg, sweetness_acidity)
    rf_model_avg = train_model(X_train_avg, y_train_avg)
    mse_avg, y_pred_avg = evaluate_model(rf_model_avg, X_test_avg, y_test_avg)
    print("Mean Squared Error using average spectral values:", mse_avg)

    # Train and evaluate using first gradients
    X_train_grad, X_test_grad, y_train_grad, y_test_grad = split_data(X_grad, sweetness_acidity)
    rf_model_grad = train_model(X_train_grad, y_train_grad)
    mse_grad, y_pred_grad = evaluate_model(rf_model_grad, X_test_grad, y_test_grad)
    print("Mean Squared Error using first gradients:", mse_grad)

    # Train and evaluate using second gradients
    X_train_second_grad, X_test_second_grad, y_train_second_grad, y_test_second_grad = split_data(X_second_grad, sweetness_acidity)
    rf_model_second_grad = train_model(X_train_second_grad, y_train_second_grad)
    mse_second_grad, y_pred_second_grad = evaluate_model(rf_model_second_grad, X_test_second_grad, y_test_second_grad)
    print("Mean Squared Error using second gradients:", mse_second_grad)

    print("Predictions using average spectral values:")
    print_predictions(y_test_avg, y_pred_avg)
    print("Predictions using first gradients:")
    print_predictions(y_test_grad, y_pred_grad)
    print("Predictions using second gradients:")
    print_predictions(y_test_second_grad, y_pred_second_grad)

if __name__ == "__main__":
    main()