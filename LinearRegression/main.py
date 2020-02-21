import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegression


def calculate_mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


if __name__ == '__main__':
    data = make_regression(n_samples=200, n_features=1, n_targets=1, random_state=42)
    X, y = data[0], data[1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    regressor = LinearRegression(lr=0.01, n_iters=500)
    regressor.fit(X_train, Y_train)

    predictions = regressor.predict(X_test)

    mse = calculate_mean_squared_error(Y_test, predictions)
    print("The mean squared error of the regressor is: {}".format(mse))

    y_pred_line = regressor.predict(X)

    assert X.shape[1] == 1
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], Y_train, color=cmap(0.9), s=10)
    plt.scatter(X_test[:, 0], Y_test, color=cmap(0.5), s=10)
    plt.plot(X[:, 0], y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()