import numpy as np
from scripts.helpers import *
from scipy.special import expit

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

def logistic_regression(y, tx, initial_w, max_iters, gamma, debug = True):
    # init parameters
    threshold = 1e-8

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        sigma_logits = expit(tx @ w)
        grad = - tx.T @ (np.multiply(y, 1 - sigma_logits) - np.multiply(1 - y, sigma_logits))
        loss = - (y.T @ np.log(sigma_logits)) - ((1 - y.T) @ np.log(1 - sigma_logits))
        w -= gamma * grad

        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if debug and iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        sigma_logits = expit(tx @ w)
        grad = - tx.T @ (np.multiply(y, 1 - sigma_logits) - np.multiply(1 - y, sigma_logits)) + 2 * lambda_ * w
        loss = - (y.T @ np.log(sigma_logits)) - ((1 - y.T) @ np.log(1 - sigma_logits)) + lambda_ * (w.T @ w)
        w -= gamma * grad

        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if debug and iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    return loss, w
