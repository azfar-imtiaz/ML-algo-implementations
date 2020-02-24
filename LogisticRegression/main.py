import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegression


def compute_accuracy(y_true, y_pred):
    return np.sum(y_pred == y_true)/y_true.shape[0]


if __name__ == '__main__':
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    clf = LogisticRegression(lr=0.01, n_iters=1000)
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)
    clf_accuracy = compute_accuracy(Y_test, predictions)

    print("The accuracy of the model is: {}".format(clf_accuracy))