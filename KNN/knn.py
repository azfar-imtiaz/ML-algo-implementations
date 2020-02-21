import numpy as np
from collections import Counter


# Euclidean distance is perhaps the most popular distance metric - however, it is not the only one that is used for KNN
# We can also use other distance metrics here 
def calculate_euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x-y)))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # there is no learning step involved in KNN. Here we simply store all training instances and their labels
        # these training instances and labels are used at prediction time
        self.X_train = X
        self.y = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calculate distances between x and all training points
        distances = [calculate_euclidean_distance(x, elem) for elem in self.X_train]

        # sort the distances, get the top k closest element indices
        indices = np.argsort(distances)
        k_indices = indices[:self.k]
        
        # get the labels of the top k closest elements
        k_labels = [self.y[idx] for idx in k_indices]

        # get most common label
        counter = Counter(k_labels)
        most_common_label = counter.most_common(n=1)[0][0]
        return most_common_label