import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        '''
            The Perceptron is a linear model that works very similar to Logistic
            Regression. The main difference in fact, between the very basic, 
            barebones implementation of the Perceptron and Logistic Regression,
            is in the activation function.
        '''
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''
            In the fit function of the Perceptron, the learning basically involves
            updating the values of the weight vector. To do this, for self.n_iters,
            we go through each instance in the training data, make a forward pass
            through the Perceptron with it (this involves getting the linear output
            and passing it through an activation function), calculate delta_w and 
            delta_b, and add them to w and b.
            In this regard, the training is quite similar to logistic regression - 
            however, the main difference is in the activation function that is used.
            The perceptron uses a unit step function as the activation function, 
            which simply returns 1 if the linear output is >= 0, else 0. Thus, the
            threshold is 0. 
            NOTE: If we use Sigmoid as the activation function in this network 
            instead, then the Perceptron becomes the same as Logistic Regression.
            NOTE: A perceptron with one or more hidden layers and a Sigmoid as the 
            final activation function is similar to stacking multiple Logistic 
            Regression models together, but perhaps not exactly the same.
        '''
        num_elems, num_features = X.shape

        # initialize the weight vector and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # This part can be improved by using Stochastic Gradient Descent, wherein
            # we would select a random element (or random batch, in case of mini-batch 
            # SGD) from the training set, and use that for updating the weights, 
            # instead of the entire training data.
            for index, elem in enumerate(X):
                # this is the forward pass
                linear_output = np.dot(elem, self.weights) + self.bias
                y_pred = self._activation(linear_output)

                # This is the weights and bias update part - the back propogation
                # The update factor is learning rate * difference between actual result
                # and predicted result.
                # For updating the weight vector, we also multiply the update factor 
                # with the input element
                # For updating the bias, we just use the update factor as is (technically,
                # we multiple it with 1 - derivating on the basis of w and b, remember?)

                update_factor = self.lr * (y[index] - y_pred)
                delta_w = update_factor * elem
                delta_b = update_factor
                self.weights += delta_w
                self.bias += delta_b

    def predict(self, X):
        '''
            In the predict function, we essentially just do a "forward pass"
            through the Perceptron, and return the result. This "forward pass"
            involves multiplying the input with the weight vector and adding the
            bias, and then applying the activation function.
        '''
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation(linear_output)
        return y_pred

    def _activation(self, x):
        '''
            np.where is like a list comprehension in Python, except that it works
            for both single elements and lists.
            This can also be thought of as a tertiary operator: first we have
            the condition, what it should return if True, and what it should
            return if False
        '''
        return np.where(x >= 0, 1, 0)