import numpy as np

def accuracy_score(y_true, y_predict):
    return sum(y_true == y_predict) / len(y_true)

def MSE(y_true, y_predict):
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

def RMSE(y_true, y_predict):
    return np.sqrt(MSE(y_true, y_predict))

def MAE(y_true, y_predict):
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r_square_score(y_true, y_predict):
    return 1 - MSE(y_true, y_predict) / np.var(y_true)