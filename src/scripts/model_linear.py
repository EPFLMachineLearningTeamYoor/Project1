import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""

    N = y.shape[0]
    e = y - tx @ w
    return(- 1. / N * e.T @ tx).T

def compute_loss(y, tx, w):
    """Calculate the loss."""

    N = y.shape[0]
    e = y - tx @ w

    return 1. / (2 * N) * (e.T @ e)

def compute_accuracy_loss(y, tx, w, lambda_ = None):
    return -np.mean(np.equal((tx @ w) > 0, y > 0))

def compute_loss_reg(y, tx, w, lambda_):
    """Calculate the loss."""

    return compute_loss(y, tx, w) + lambda_ * w.T @ w
