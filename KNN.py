import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def _predict(self, x):
        distance = [np.sqrt((x_train - x) ** 2) for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        vote = Counter(topK_y)

        return vote.most_common(1)[0][0]

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)