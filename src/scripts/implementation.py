import numpy as np
from scripts.helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma, debug = False):
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, debug)
    return ws[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, debug = False)
    return ws[-1]

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # returns optimal weights
    # ***************************************************

    N = tx.shape[0]
    w = np.linalg.pinv(tx.T @ tx) @ tx.T @ y
    e = y - tx @ w
    return w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = tx.shape[0]
    w = np.linalg.pinv(tx.T @ tx + lambda_ * (2. * N) * np.eye(tx.shape[1])) @ tx.T @ y
    e = y - tx @ w
    return w
