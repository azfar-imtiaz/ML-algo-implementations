import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from perceptron import Perceptron


def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_pred)


if __name__ == '__main__':
    X, y = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    # classes should be 1 or 0 for the Perceptron
    y = [1 if a > 0 else 0 for a in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = Perceptron(learning_rate=0.01, n_iters=200)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = compute_accuracy(y_test, predictions)

    print("The accuracy of the model is: {}".format(accuracy))

    # plotting the decision boundary
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)

    x0_1 = np.amin(X_train[:,0])
    x0_2 = np.amax(X_train[:,0])

    x1_1 = (-clf.weights[0] * x0_1 - clf.bias) / clf.weights[1]
    x1_2 = (-clf.weights[0] * x0_2 - clf.bias) / clf.weights[1]

    ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k')

    y_min = np.amin(X_train[:,1])
    y_max = np.amax(X_train[:,1])
    ax.set_ylim([y_min-3, y_max+3])

    plt.show()