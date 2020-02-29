import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self._priors = None
        self._means = None
        self._vars = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._classes = list(set(y))
        num_classes = len(self._classes)

        # these all can be dictionaries too
        # self._priors = np.zeros(num_samples, dtype=np.float)
        # self._means = np.zeros((num_classes, num_features), dtype=np.float)
        # self._vars = np.zeros((num_classes, num_features), dtype=np.float)
        self._priors = dict()
        self._means = dict()
        self._vars = dict()

        for cl in self._classes:
            # first, get all records belonging to this class
            X_cls = X[cl == y]
            # calculate the mean, variance and prior for this class
            prior = X_cls.shape[0] / y.shape[0]
            # the axis here is 0 because we have to compute the mean/variance of all the features
            # of the data, per class. Therefore, we compute the mean along axis 0, which is the rows
            mean = np.mean(X_cls, axis=0)
            var = np.var(X_cls, axis=0)

            self._priors[cl] = prior
            # self._means[cl, :] = mean
            # self._vars[cl, :] = var
            self._means[cl] = mean
            self._vars[cl] = var

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = np.zeros(len(self._classes))

        for cl in self._classes:
            prior = self._priors[cl]
            
            # compute the class conditional probability through the probability density function - gaussian, in this case
            mean = self._means[cl]
            var = self._vars[cl]
            numerator = np.exp(-(x - mean)**2 / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            cls_conditional_prob = numerator / denominator

            posterior = np.sum(np.log(prior + cls_conditional_prob))
            posteriors[cl] = posterior

        return np.argmax(posteriors)