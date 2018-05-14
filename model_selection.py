import numpy as np

def train_test_split(X, y, test_ratio = 0.2, seed = None):
    if seed:
        np.random.seed(seed)
    
    shuffled_index = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_index = shuffled_index[test_size:]
    test_index = shuffled_index[:test_size]

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    return X_train, X_test, y_train, y_test
    
    