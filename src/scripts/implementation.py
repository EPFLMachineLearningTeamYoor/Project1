import numpy as np
from scripts.helpers import *
import scripts.model_linear as model_linear
import scripts.model_logistic as model_logistic

def get_last(ws, losses):
    return (ws[-1], np.mean(losses[-1]))

def least_squares_GD(y, tx, initial_w, max_iters, gamma, debug = False):
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_linear.compute_loss, grad_f = model_linear.compute_gradient, debug = debug)
    return get_last(ws, losses)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, debug = False):
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma, loss_f = model_linear.compute_loss, grad_f = model_linear.compute_gradient, debug = debug)
    return get_last(ws, losses)

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # returns optimal weights
    # ***************************************************

    N = tx.shape[0]
    w = np.linalg.pinv(tx.T @ tx) @ tx.T @ y
    e = y - tx @ w
    loss = 1. / 2 / N * e.T @ e
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = tx.shape[0]
    w = np.linalg.pinv(tx.T @ tx + lambda_ * (2. * N) * np.eye(tx.shape[1])) @ tx.T @ y
    e = y - tx @ w
    loss = 1./ 2 / N * e.T @ e + lambda_ * w.T @ w
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, debug = False):
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_logistic.loss, grad_f = model_logistic.grad, debug = debug)
    return get_last(ws, losses)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, debug = False):
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_logistic.reg_loss, grad_f = model_logistic.reg_grad, kwargs = {'lambda_': lambda_}, debug = debug)
    return get_last(ws, losses)
