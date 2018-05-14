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

    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)
