import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=20):
        self.lr = lr
        self.n_iters = n_iters
        # the weight matrix is initially set to None because we do not know the shape - not until we get the training data
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        # get the number of features from training data. This will be used to initialize the weight vector
        num_features = X.shape[1]
        # this is a weight vector, and not a matrix. Can be initialized with random values too
        self.weight = np.zeros(num_features)
        # the bias is always a singular value
        self.bias = 0

        # The learning process runs for a fixed number of iterations
        # This is not stochastic gradient descent - otherwise, we would choose a subset of the data instead of the whole data
        # in each iteration to update the model parameters weight and b
        for i in range(self.n_iters):

            # In each iteration, we calculate the derivate of the cost function (with respect to weight vector and bias separately)
            # Using these derivatives, we update the weight vector and the bias, and that's how the "learning" happens
            # Therefore, the calculation happens on the basis of the cost function
            # To make it truly generic, we should specify the cost function when initializing the regressor, 
            # and calculate/use the derivation of that cost function accordingly

            # These four lines here calculate the derivative of the cost function (MSE) with respect to weight vector and bias
            # This part should technically happen in a loop, since all training instances are being used here
            # However, we can use matrix operations to get around that, and be more efficient in the process
            y_pred = np.dot(X, self.weight) + self.bias
            diff = (y_pred - y)
            # Remember: The dot operation multiplies the corresponding elements of two matrices and adds them to return a single number
            dw = (1/X.shape[0]) * np.dot(X.T, diff)
            db = (1/X.shape[0]) * np.sum(diff)
            
            # this here is technically the correct calculation of dw and db; however, 2 is a constant here that is typically ignored
            # dw = (1/X.shape[0]) * 2 * np.dot(X.T, diff)
            # db = (1/X.shape[0]) * 2 * np.sum(diff)

            # Here, the weight vector and bias are updated using dw and db calculated above
            # This is gradient descent in action here. The fact that we are subtracting the derivative from the original term
            # means that we are moving in the direction opposite to the slope - gradient "descent"
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        # this is essentially the same as: y = wx + b
        pred = np.dot(X, self.weight) + self.bias
        return pred