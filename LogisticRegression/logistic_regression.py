import numpy as np
from scipy.special import expit


class LogisticRegression:
    '''
        TODO: Add predict_proba function in this class, and then calculate ROC AUC score in main!
        
        The main difference between Linear Regression and Logistic Regression (in terms of implementation) is that in
        Logistic Regression, we apply a Sigmoid to the output of the linear model to get a prediction probability. The 
        cost function in this case is different too - now it's Cross Entropy instead of Mean Squared Error.
    '''
    def __init__(self, lr, n_iters):
        '''
            This part is pretty much the same as Linear Regression - we just add a classification probability threshold.
        '''
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
        self.clf_threshold = 0.5

    def fit(self, X, y):
        '''
            Use normal gradient descent to learn the parameters of the model - which in case are the weight vector and 
            the bias.
        '''
        num_features = X.shape[1]
        self.weight = np.zeros(num_features)
        self.bias = 0

        for i in range(self.n_iters):
            # apply the same line function formula from linear regression to get the linear model output
            linear_model_output = np.dot(X, self.weight) + self.bias
            # apply the sigmoid on the linear model output to get a prediction probability
            y_pred = self._sigmoid(linear_model_output)

            # Just like before, the result of the derivation of the cost function w.r.t both w and b includes a 2 in the 
            # equation - however, it is regarded as a scaling factor and thus ignored.
            diff = y_pred - y
            dw = (1/X.shape[0]) * np.dot(X.T, diff)
            db = (1/X.shape[0]) * np.sum(diff)

            # Gradient descent in action here, same as before
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        '''
            At the time of prediction, we apply the same functions as before to get y_pred. However, we define a threshold
            to compare it against in order to identify which class the given test instance belongs to.
        '''
        linear_model_output = np.dot(X, self.weight) + self.bias
        y_pred = self._sigmoid(linear_model_output)
        y_pred_cls = [1 if pred >= self.clf_threshold else 0 for pred in y_pred]
        return y_pred_cls

    def _sigmoid(self, x):
        # this is giving an overflow warning, so I used Scipy's version of the logistic sigmoid function
        # return 1 / (1 + np.exp(-x))
        return expit(x)