import numpy as np
import csv

def get_header(path):
    header = None
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            header = row[2:]
            break
    return header

def impute_with_mean(X, missing_val = -999):
    X = np.copy(X)
    X[X == missing_val] = None
    nan_mean = np.nanmean(X, axis = 0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(nan_mean, inds[1])
    return X


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def compute_gradient(y, tx, w):
    """Compute the gradient."""

    N = y.shape[0]
    e = y - tx @ w
    return(- 1. / N * e.T @ tx)

def gradient_descent(y, tx, initial_w, max_iters, gamma, debug = True):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient
        ws.append(w)
        losses.append(loss)
        if debug:
          print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_loss(y, tx, w):
    """Calculate the loss."""

    N = y.shape[0]
    e = y - tx @ w

    return 1. / (2 * N) * (e.T @ e)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return gd_m.compute_gradient(y, tx, w)

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, debug = False):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for (n_iter, (y_, tx_)) in enumerate(batch_iter(y, tx, batch_size = batch_size,
                                                    num_batches=max_iters, shuffle=True)):
        loss = compute_loss(y, tx, w)
        stoch_gradient = compute_stoch_gradient(y_, tx_, w)
        w -= gamma * stoch_gradient
        ws.append(w)
        losses.append(loss)
        if Debug:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
