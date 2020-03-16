import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self._priors = None
        self._means = None
        self._vars = None
        self._classes = None

    def fit(self, X, y):
        '''
            In the fit function of Naive Bayes, we calculate the prior probability, the mean
            and the variance for each class separately. This defines the probabilistic nature
            of the Naive Bayes classifier, and allows us to compute the label and probability
            of a given instance at test time.
        '''
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

        # for each class, we calculate the prior, mean and variance
        # the mean and variance are calculated across all features
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
        '''
            At test time, we predict the label of each test instance, along with its
            probability. For a given test instance, we calculate the probability for
            each class separately, and return the class with the highest probability.
            This probability that we calculate at test time is the posterior probability,
            and is calculated by using the priors we computed using the training data,
            and calculating the likelihood of each class. This likelihood is computed
            using the Probability Density Function.
        '''
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # posteriors = np.zeros(len(self._classes))
        posteriors = dict()

        # We calculate the posterior probability of each class using the Naive Bayes formula
        # The Naive Bayes formula is: (Likelihood x Prior) / Evidence
        # We ignore the Evidence in Naive Bayes, so we calculate the Likelihood, and use the prior calculated above
        # The evidence is P(x|c), and it is calculated using a PDF (Probability Density Function) for each class separately
        for cl in self._classes:
            prior = self._priors[cl]

            # compute the class conditional probability through the PDF - Gaussian, in this case
            mean = self._means[cl]
            var = self._vars[cl]
            numerator = np.exp(-(x - mean)**2 / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            likelihood = numerator / denominator

            # the log is taken here to simplify the calculations
            posterior = np.sum(np.log(prior + likelihood))
            posteriors[cl] = posterior

        # return np.argmax(posteriors)
        return max(posteriors, key=lambda k: posteriors[k])
