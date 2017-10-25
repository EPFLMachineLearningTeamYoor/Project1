import numpy as np
from scipy.special import expit
from scipy.sparse import diags

def grad(y, tx, w):
    """ returns logistic regression gradient """
    (N, D) = tx.shape
    return (tx.T @ (expit(tx @ w) - y)).reshape(D, 1)

def loss(y, tx, w):
    """ returns logistic regression loss """
    return -np.sum(np.multiply((tx @ w).flatten(), y.flatten())) + np.sum(np.log1p(np.exp(tx @ w)))

def reg_grad(y, tx, w, lambda_):
    """ returns regularized logistic regression gradient """
    (N, D) = tx.shape
    return grad(y, tx, w) + (2 * lambda_ * w).reshape(D, 1)

def reg_loss(y, tx, w, lambda_):
    """ returns regularized logistic regression loss """
    return loss(y, tx, w) + lambda_ * (w.T @ w)

def newton_grad(y, x, w, lambda_ = 0):
    """ returns newton gradient """
    N, D = x.shape
    sigma = expit(x @ w).flatten()
    S = diags(np.multiply(sigma, 1 - sigma))
    H = x.T @ S @ x
    assert H.shape == (D, D), "H shape"
    return np.linalg.pinv(H + np.eye(D) * 2 * lambda_) @ reg_grad(y, x, w, lambda_)

def newton_reg_grad(y, x, w, lambda_):
    """ returns regularized newton gradient """
    return newton_grad(y, x, w, lambda_)

### SECONDARY IMPLEMENTATION

def sigmoid(t):
    numerator = np.exp(t)
    denominator = np.add(1,numerator)
    return (numerator/denominator)

def calculate_loss(y, tx, w):
    return -(np.mean(y.T @ tx @ w) - np.mean(np.sum(np.log(1 + np.exp(tx @ w)))))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    step1 = sigmoid(tx.dot(w)) - y
    step2 = tx.T
    gradient = np.dot(step2,step1)
    return gradient

### TESTS SECTION

def get_random_data(N = 10000, D = 100, seed = 1):
#    N = 250000
#    D = 73
    N = 100
    D = 5
    np.random.seed(seed)
    x = np.random.randn(N, D)
    y = np.random.rand(N, 1) > 0.5
    w = np.random.randn(D, 1)
    return x, y, w

def test_loss_1(seed = 1):
    x, y, w = get_random_data(seed = seed)
    [x_, y_, w_] = [np.copy(z) for z in (x, y, w)]
    assert np.allclose(calculate_loss(y, x, w), loss(y, x, w)), "log_reg_loss"
    for z, z_ in zip((x, y, w), (x_, y_, w_)):
        assert np.allclose(z, z_), "copy"

def test_grad_1(seed = 1):
    x, y, w = get_random_data(seed = seed)
    [x_, y_, w_] = [np.copy(z) for z in (x, y, w)]
    assert np.allclose(calculate_gradient(y, x, w), grad(y, x, w)), "log_reg_grad"
    for z, z_ in zip((x, y, w), (x_, y_, w_)):
        assert np.allclose(z, z_), "copy"

def test_all():
    seeds = [9,2,3,42,56]
    for s in seeds:
        test_grad_1(seed = s)
    for s in seeds:
        test_loss_1(seed = s)
    return 1

if __name__ == "__main__":
    if test_all():
        print("TESTS PASSED")
