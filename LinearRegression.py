import numpy as np


class LinearRegression:

    def __init__(self):
        self.interception = None
        self.coefficient = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception = self._theta[0]
        self.coefficient = self._theta[1:]

        return self
    
    # Gradient Descent
    def fit_gd(self, X_train, y_train, eta=1e-4, n_iters=10000):

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
        
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            
            # return res * 2 / len(X_b)

            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=10000, epsilon=1e-6):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters=10000)
        self.interception = self._theta[0]
        self.coefficient = self._theta[1:]

    # Stochastic Gradient Descent    
    def fit_sgd(self, X_train, y_train, a=10, b=100, n_iters=10):

        def dJ_sgd(theta, X_b_i, y_i): # for only one data
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2

        def sgd(X_b, y, initial_theta, n_iters, a=10, b=100):

            def learning_rate(t):
                return a / (t + b)
            
            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                # make sure all the instances have been used
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient
            
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.random(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, a, b)
        self.interception = self._theta[0]
        self.coefficient = self._theta[1:]


    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)
