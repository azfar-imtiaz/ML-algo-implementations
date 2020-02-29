import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from naive_bayes import NaiveBayesClassifier


def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    clf = NaiveBayesClassifier()
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)

    accuracy = compute_accuracy(Y_test, predictions)
    print("The accuracy of the model is: {}".format(accuracy))