import numpy as np

def binarize_categorical_feature(f):
    values = sorted(list(set(f[:,0])))
    x = np.zeros((f.shape[0], 1))
    for v in values:
        x = np.hstack((x, f == v))
    return x[:,1:]

def binarize_categorical(x, ids):
    x_ = np.zeros((x.shape[0], 1))
    for idx in ids:
        x_ = np.hstack((x_, binarize_categorical_feature(x[:, idx:idx+1])))
    x = np.delete(x, ids, axis=1)
    x = np.hstack((x, x_[:, 1:]))
    return x

def impute_with_mean(X_, ids, missing_val = -999):
    X_ = np.copy(X_)
    X = X_[:, ids]
    X[X == missing_val] = None
    nan_mean = np.nanmean(X, axis = 0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(nan_mean, inds[1])
    X_[:, ids] = X
    return X_

def add_polynomial(X, ids, max_degrees = 2):
    if type(max_degrees) == type(int):
        max_degrees = np.ones((len(ids))) * max_degrees
    X_orig = X
    X = np.copy(X)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    for i, idx in enumerate(ids):
        for degree in range(2, max_degrees[i] + 1):
            X = np.hstack((X, np.power(X_orig[:, idx:idx+1], degree)))
    return X

def indicator_missing(X, ids, missing_val = -999.):
    X = np.copy(X)
    for idx in ids:
        f_miss = 1. * (X[:, idx:idx + 1] == missing_val)
        X = np.hstack((X, f_miss))
    return X

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

def test_missing_1():
    X = [[0,1,2],[3,-999,5],[0,3,-999]]
    X_copy = np.copy(X)
    X_ans = np.array([[   0.,    1.,    2.,    0.,    0.],
           [   3., -999.,    5.,    1.,    0.],
           [   0.,    3., -999.,    0.,    1.]])
    assert np.allclose(X_ans, indicator_missing(X, [1,2], missing_val = -999.)), "indicator_missing"
    assert np.all(X_copy == X), "copy"

def test_all():
    test_binarize_1()
    test_binarize_2()
    test_impute_1()
    test_poly_1()
    test_missing_1()
    return 1

if __name__ == "__main__":
    if test_all():
        print("Tests passed")
