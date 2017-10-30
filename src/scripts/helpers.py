from tqdm import tqdm
import numpy as np
import csv
from scripts import model_linear
from matplotlib import pyplot as plt

def get_last_ans(ws, losses):
    """ return last w in array and last loss in array """
    return (ws[-1], np.mean(losses[-1]))

def y_to_01(y):
    """ convery y in -1,1 to y in 0,1 for logistic regression """
    y = np.copy(y)
    y[y[:,0] == -1,0] = 0
    return y

def get_header(path):
    """ get feature names from a file """
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
    ws, losses = [initial_w], []

    acc_arr = []
    loss_arr = []

    # Parameter
    w = np.copy(initial_w)

    with tqdm(total = max_iters, unit = 'epoch') as pbar:
      for n_iter in range(max_iters):
        # calculating loss and gradient
        loss = loss_f(y, tx, w, **kwargs)
        gradient = grad_f(y, tx, w, **kwargs)

        # updating w
        w -= gamma * gradient

        # adding w, loss
        [x.append(y) for x, y in zip((ws, losses), (w, loss))]

        # debug message print
        if debug == True:
            print("Gradient Descent(%d/%d): loss=%.2f grad_norm=%.2f w_norm=%.2f" % (n_iter, max_iters - 1, np.mean(loss), np.linalg.norm(gradient), np.linalg.norm(w)))

        acc_arr.append(model_linear.compute_accuracy_loss(y, tx,  w))
        loss_arr.append(loss)
        pbar.set_postfix(loss=round(np.mean(loss), 2), grad=round(np.linalg.norm(gradient), 2), w=round(np.linalg.norm(w), 2), acc = round(model_linear.compute_accuracy_loss(y, tx,  w), 2))
        pbar.update(1)

    if debug == 'plot':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Iteration')
        ax.plot(range(len(loss_arr)), loss_arr, label='loss', c='green')
        ax.set_ylabel('Loss')
        ax2 = ax.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.plot(range(len(acc_arr)), acc_arr, label='accuracy', c='red')
        ax.legend(loc=2)
        ax2.legend(loc=0)
        plt.show()

    return losses, ws

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, loss_f, grad_f, kwargs = {}, debug = False):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws, losses = [initial_w], []

    max_up_count = 10

    # Parameter
    w = np.copy(initial_w)

    ls_n_b = []
    w_n_b = []
    g_n_b = []

    acc_arr = []
    loss_arr = []

    NORM_MAX = 1e7

    w_old = np.copy(w)

    with tqdm(total = max_iters, unit = 'epoch') as pbar:
      for (n_iter, (overlap, y_, tx_)) in enumerate(batch_iter(y, tx, batch_size = batch_size, num_batches=max_iters, shuffle=True)):
        # calculating loss and gradient
        loss = loss_f(y, tx, w, **kwargs)
        stoch_gradient = grad_f(y_, tx_, w, **kwargs)

        # updating w
        w -= gamma * stoch_gradient

        if np.linalg.norm(w) > NORM_MAX:
            w = w / np.linalg.norm(w) * NORM_MAX

        # adding w, loss
        [x.append(y) for x, y in zip((ws, losses), (w, loss))]
        [x.append(np.linalg.norm(y)) for x, y in zip((ls_n_b, w_n_b, g_n_b), (loss, w, stoch_gradient))]

        # debug message print
        if debug == True:
            print("Stochastic Gradient Descent(%d/%d): loss=%.2f" % (n_iter, max_iters - 1, np.mean(loss)))

        if overlap and len(g_n_b) > 0:
            eps_w = np.linalg.norm(w - w_old) / np.mean(w_n_b) / gamma
#            if eps_w < 1e-2:
#                return losses[:-1], ws[:-1]
            pbar.set_postfix(loss=round(np.mean(ls_n_b), 2), grad=round(np.mean(g_n_b), 2), w=round(np.mean(w_n_b), 2), acc = round(model_linear.compute_accuracy_loss(y, tx,  w), 2), diff=eps_w)
            acc_arr.append(model_linear.compute_accuracy_loss(y, tx,  w))
            loss_arr.append(np.mean(ls_n_b))
            ls_n_b, w_n_b, g_n_b = [], [], []
            w_old = np.copy(w)

        pbar.update(1)

    if debug == 'plot':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Iteration')
        ax.plot(range(len(loss_arr)), loss_arr, label='loss', c='green')
        ax.set_ylabel('Loss')
        ax2 = ax.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.plot(range(len(acc_arr)), acc_arr, label='accuracy', c='red')
        ax.legend(loc=2)
        ax2.legend(loc=0)
        plt.show()

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
    start_index = 0

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        overlap = start_index > data_size
        if overlap:
            start_index = 0
        end_index = min(start_index + batch_size, data_size)
        if start_index != end_index:
            yield overlap, shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
        start_index += batch_size
