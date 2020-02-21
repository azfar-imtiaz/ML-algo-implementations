from knn import KNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data = load_iris()
    X, y = data.data, data.target

    print("Total unique classes in dataset: {}".format(len(set(y))))
    print("Unique classes in dataset are: {}".format(list(set(y))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    clf = KNN(k=5)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)

    print("Accuracy of the KNN classifier is: {}".format(accuracy))

    print("Plotting the dimensions of the data using the first two dimensions")
    plt.scatter(np.array(X_train[:,0]), np.array(X_train[:, 1]), c=y_train)
    plt.scatter(np.array(X_test[:,0]), np.array(X_test[:, 1]), c=predictions, marker='x')
    plt.show()