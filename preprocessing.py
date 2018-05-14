import numpy as np


class StanardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self
    
    def transform(self, X):
        res = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            res[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return res
