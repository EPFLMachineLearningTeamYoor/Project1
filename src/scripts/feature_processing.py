from tqdm import tqdm
import numpy as np

def standardize(x, std_x = None, mean_x = None, ignore_first = True):
    """Standardize the original data set."""
    x = np.copy(x)
    if type(mean_x) == type(None):
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if ignore_first:
        x[:,0] = 1
    if type(std_x) == type(None):
        std_x = np.std(x, axis=0)
    for i in range(std_x.shape[0]):
        if std_x[i] > 0: x[:, i] = x[:, i] / std_x[i]
    return x, mean_x, std_x

def binarize_categorical_feature(f):
    """ return binary columns for each feature value """
    values = sorted(list(set(f[:,0])))
    assert len(values) < 10, "too many categories"
    x = np.zeros((f.shape[0], 1))
    for v in values:
        x = np.hstack((x, f == v))
    return x[:,1:]

def binarize_categorical(x, ids):
    """ replace categorical feature with multiple binary ones """
    x_ = np.zeros((x.shape[0], 1))
    for idx in ids:
        x_ = np.hstack((x_, binarize_categorical_feature(x[:, idx:idx+1])))
    x = np.delete(x, ids, axis=1)
    x = np.hstack((x, x_[:, 1:]))
    return x

def impute_with_mean(X_, ids, missing_val = -999):
    """ replace missing_val with mean value on columns ids """
    X_ = np.copy(X_)
    X = X_[:, ids]
    X[X == missing_val] = None
    nan_mean = np.nanmean(X, axis = 0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(nan_mean, inds[1])
    X_[:, ids] = X
    return X_

def add_polynomial(X, ids, max_degrees = 2):
    """ add constant feature and degrees of features ids up to selected degree """
    if type(max_degrees) == int:
        max_degrees = [max_degrees] * len(ids)
    X_orig = X
    X = np.copy(X)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    for i, idx in enumerate(ids):
        for degree in range(2, max_degrees[i] + 1):
            X = np.hstack((X, np.power(X_orig[:, idx:idx+1], degree)))
    return X

def indicator_missing(X, ids, missing_val = -999.):
    """ add binary feature indicating if original feature was missing """
    X = np.copy(X)
    for idx in ids:
        f_miss = 1. * (X[:, idx:idx + 1] == missing_val)
        X = np.hstack((X, f_miss))
    return X

def add_mult(X):
    n = X.shape[1]
    res = np.hstack((np.copy(X), np.zeros((X.shape[0], n * (n - 1)))))
    k = n
    pbar = tqdm(total=n * (n - 1))
    for i in range(n):
        for j in range(i + 1, n):
            res[:, k] = np.multiply(X[:, i], X[:, j])
            k += 1
            pbar.update(1)
    return res

### DATASET-SPECIFIC FUNCTIONS

need_impute = [0, 5, 6, 12, 23, 24, 25, 26, 27, 28]
categorical = [23] #+1
need_poly = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29]

def process_X(X, degree, tpl = (None, None)):
    res = None
    (x_mean, x_std) = tpl
    with tqdm(total=6) as pbar:
        X_1 = indicator_missing(X, need_impute)
        pbar.update(1)
        X_2 = impute_with_mean(X_1, need_impute)
        pbar.update(1)
        X_22 = add_mult(X_2)
        pbar.update(1)
        X_3 = add_polynomial(X_22, need_poly, max_degrees = degree)
        pbar.update(1)
        X_4 = binarize_categorical(X_3, categorical)
        pbar.update(1)
        X_5, x_mean, x_std = standardize(X_4, mean_x = x_mean, std_x = x_std)
        pbar.update(1)
        res = X_5
    return res, (x_mean, x_std)

### TESTING SECTION

def test_binarize_1():
    x = np.array([[1,2,2,4]]).T
    x_copy = np.copy(x)
    x_ = np.array([[1,0,0],[0,1,0],[0,1,0],[0,0,1]])
    assert np.all(x_ == binarize_categorical_feature(x)), "binarize_categorical_feature"
    assert np.all(x == x_copy), "copy"

def test_binarize_2():
    x = np.array([[1,2,2,4],[0.1,0.2,0.3,0.4],[1,-1,1,1]]).T
    x_copy = np.copy(x)
    x_ = np.array([[0.1,1,0,0,0,1],[0.2,0,1,0,1,0],[0.3,0,1,0,0,1],[0.4,0,0,1,0,1]])
    assert np.all(x_ == binarize_categorical(x, [0,2])), "binarize_categorical"
    assert np.all(x == x_copy), "copy"

def test_impute_1():
    X_ = np.array([[0,1,1],[3,4,0],[0,0,-999]], dtype=np.float64)
    X_copy = np.copy(X_)
    X__ = np.array([[ 0. ,  1. ,  1. ], [ 3. ,  4. ,  0. ], [ 0. ,  0. ,  0.5]])
    assert np.all(X__ == impute_with_mean(X_, [2], missing_val=-999)), "impute_with_mean"
    assert np.all(X_ == X_copy), "copy"

def test_poly_1():
    X = np.array([[0,1,2],[3,4,5],[0.5,0.6,0.7]])
    X_copy = np.copy(X)

    X_ans = np.array([[  1.   ,   0.   ,   1.   ,   2.   ,   0.   ,   0.   ,   4.   ],
           [  1.   ,   3.   ,   4.   ,   5.   ,   9.   ,  27.   ,  25.   ],
           [  1.   ,   0.5  ,   0.6  ,   0.7  ,   0.25 ,   0.125,   0.49 ]])
    assert np.allclose(X_ans, add_polynomial(X, [0,2], max_degrees = [3,2])), "add_polynomial"
    assert np.all(X_copy == X), "copy"

def test_poly_2():
    X = np.array([[0,1,2],[3,4,5],[0.5,0.6,0.7]])
    X_copy = np.copy(X)

    X_ans = np.array([[  1.   ,   0.   ,   1.   ,   2.   ,   0.   ,     4.   ],
           [  1.   ,   3.   ,   4.   ,   5.   ,   9.   ,  25.   ],
           [  1.   ,   0.5  ,   0.6  ,   0.7  ,   0.25 ,   0.49 ]])
    assert np.allclose(X_ans, add_polynomial(X, [0,2], max_degrees = 2)), "add_polynomial"
    assert np.all(X_copy == X), "copy"

def test_missing_1():
    X = [[0,1,2],[3,-999,5],[0,3,-999]]
    X_copy = np.copy(X)
    X_ans = np.array([[   0.,    1.,    2.,    0.,    0.],
           [   3., -999.,    5.,    1.,    0.],
           [   0.,    3., -999.,    0.,    1.]])
    assert np.allclose(X_ans, indicator_missing(X, [1,2], missing_val = -999.)), "indicator_missing"
    assert np.all(X_copy == X), "copy"

def test_stand_1():
    x = [[1,2,3],[4,5,6]]
    x_copy = np.copy(x)
    x_ans = (np.array([[-1., -1., -1.], [ 1.,  1.,  1.]]), np.array([ 2.5,  3.5,  4.5]), np.array([ 1.5,  1.5,  1.5]))
    x_st = standardize(x, ignore_first = False)
    for a, b in zip(x_ans, x_st):
        assert np.allclose(a, b), "standardize"
    assert np.all(x_copy == x), "copy"

def test_all():
    test_binarize_1()
    test_binarize_2()
    test_impute_1()
    test_poly_1()
    test_poly_2()
    test_missing_1()
    test_stand_1()
    return 1

if __name__ == "__main__":
    if test_all():
        print("Tests passed")
