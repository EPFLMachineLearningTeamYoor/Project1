import numpy as np
from scripts.helpers import *
from scripts.implementation import *

def check_model(X, w, y, thr = 0.9):
    assert np.mean((y > 0) == (X @ w > 0)) > thr, "model accuracy"

def gen_data(N, D):
    X = np.hstack((np.ones((N, 1)), np.random.randn(N,D)))
    w_orig = np.array([[1], [1], [2]])
    y = 1. * ((X @ w_orig + 0.1 * np.random.rand(N, 1)) > 0) * 2 - 1
    return X, y


def test_all(N = 1000, D = 2, seed = 42):
    np.random.seed(seed)

    X, y = gen_data(N, D)
    y_01 = y_to_01(y)
    w_initial = np.random.rand(D + 1, 1)

    (w, l) = least_squares_GD(y, X, w_initial, 100, 0.01, debug = False)
    check_model(X, w, y)

    (w, l) = least_squares_SGD(y, X, w_initial, 100, 0.1)
    check_model(X, w, y)

    (w, l) = least_squares(y, X)
    check_model(X, w, y)

    (w, l) = ridge_regression(y, X, 0.1)
    check_model(X, w, y)

    (w, l) = logistic_regression(y_01, X, w_initial, 100, 0.01, debug = False)
    check_model(X, w, y_01)

    (w, l) = reg_logistic_regression(y_01, X, 1, w_initial, 100, 0.01, debug = False)
    check_model(X, w, y_01)

    return(1)

if __name__ == "__main__":
    if test_all():
        print("Tests passed")
