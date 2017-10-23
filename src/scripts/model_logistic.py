import numpy as np
from scipy.special import expit

def grad(y, tx, w):
    return tx.T @ (expit(tx @ w) - y)

def loss(y, tx, w):
    return np.sum(- np.multiply(tx @ w, y) + np.log(1 + np.exp(tx @ w)))

def reg_grad(y, tx, w, lambda_):
    return grad(y, tx, w) + 2 * lambda_ * w

def reg_loss(y, tx, w, lambda_):
    return loss(y, tx, w) + lambda_ * (w.T @ w)
