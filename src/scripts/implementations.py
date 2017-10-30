import numpy as np
from scripts.helpers import *
import scripts.model_linear as model_linear
import scripts.model_logistic as model_logistic

def least_squares_GD(y, tx, initial_w, max_iters, gamma, debug = False):
    """ implement least squares via gradient descent """
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_linear.compute_loss, grad_f = model_linear.compute_gradient, debug = debug)
    return get_last_ans(ws, losses)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, debug = False):
    """ implement least squares via stochastic gradient descent """
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma, loss_f = model_linear.compute_loss, grad_f = model_linear.compute_gradient, debug = debug)
    return get_last_ans(ws, losses)

def least_squares(y, tx):
    """calculate the least squares solution."""

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
    """ implement logistic regression via gradient descent """
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_logistic.loss, grad_f = model_logistic.grad, debug = debug)
    return get_last_ans(ws, losses)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, debug = False):
    """ implement regularized logistic regression via gradient descent """
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_logistic.reg_loss, grad_f = model_logistic.reg_grad, kwargs = {'lambda_': lambda_}, debug = debug)
    return get_last_ans(ws, losses)

def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma, debug = False):
    """ implement regularized logistic regression via Newton method """
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma, loss_f = model_logistic.reg_loss, grad_f = model_logistic.newton_reg_grad, kwargs = {'lambda_': lambda_}, debug = debug)
    return get_last_ans(ws, losses)

def reg_logistic_regression_batch(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, debug = False):
    """ implement regularized logistic regression via SGD """
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_f = model_logistic.reg_loss, grad_f = model_logistic.reg_grad, kwargs = {'lambda_': lambda_}, debug = debug)
    return get_last_ans(ws, losses)

def reg_logistic_regression_newton_batch(y, tx, lambda_, initial_w, batch_size, max_iters, gamma, debug = False):
    """ implement regularized logistic regression via SGD and Newton """
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_f = model_logistic.reg_loss, grad_f = model_logistic.newton_reg_grad, kwargs = {'lambda_': lambda_}, debug = debug)
    return get_last_ans(ws, losses)

### TESTING SECTION

def check_model(X, w, y, thr = 0.9):
    """ function which checks if > thr accuracy """
    assert np.mean((y > 0) == (X @ w > 0)) > thr, "model accuracy"

def gen_data(N, D):
    """ generate some random data """
    X = np.hstack((np.ones((N, 1)), np.random.randn(N,D)))
    w_orig = np.array([[1], [1], [2]])
    y = 1. * ((X @ w_orig + 0.1 * np.random.rand(N, 1)) > 0) * 2 - 1
    return X, y


def test_all(N = 1000, D = 2, seed = 42):
    """ test all methods """
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

    (w, l) = reg_logistic_regression_newton(y_01, X, 1, w_initial, 100, 0.01, debug = False)
    check_model(X, w, y_01)

    (w, l) = reg_logistic_regression_newton_batch(y_01, X, 1, w_initial, 10, 100, 0.01, debug = False)
    check_model(X, w, y_01)

    return(1)

if __name__ == "__main__":
    if test_all():
        print("Tests passed")
