import numpy as np
import csv

def get_last_loss(ws, losses):
    return (ws[-1], np.mean(losses[-1]))

def y_to_01(y):
    y = np.copy(y)
    y[y[:,0] == -1,0] = 0
    return y

def get_header(path):
    header = None
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            header = row[2:]
            break
    return header

def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f, grad_f, kwargs = {}, debug = True):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = loss_f(y, tx, w, **kwargs)
        gradient = grad_f(y, tx, w, **kwargs)
        w -= gamma * gradient
        ws.append(w)
        losses.append(loss)
        if debug:
            print("Gradient Descent(%d/%d): loss=%.2f grad_norm=%.2f w_norm=%.2f" % (n_iter, max_iters - 1, np.mean(loss), np.linalg.norm(gradient), np.linalg.norm(w)))

    return losses, ws

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, loss_f, grad_f, kwargs = {}, debug = False):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for (n_iter, (y_, tx_)) in enumerate(batch_iter(y, tx, batch_size = batch_size,
                                                    num_batches=max_iters, shuffle=True)):
        loss = loss_f(y, tx, w, **kwargs)
        stoch_gradient = grad_f(y_, tx_, w, **kwargs)
        w -= gamma * stoch_gradient
        ws.append(w)
        losses.append(loss)
        if debug:
            print("Stochastic Gradient Descent(%d/%d): loss=%.2f" % (n_iter, max_iters - 1, np.mean(loss)))

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
