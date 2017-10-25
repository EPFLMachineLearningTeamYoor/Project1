import numpy as np
from matplotlib import pyplot as plt
from scripts import plots
from tqdm import tqdm

def build_k_indices(num_row, k_fold, seed):
    """build and seperate k random indices for k-fold."""
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, model, kw_model, loss, kw_loss, lambda_):
    """return the loss of ridge regression."""

    # indices for test.
    idx_te = k_indices[k]
    
    # indices for train.Convert also 1d array by merging N-1 folds.
    idx_tr = k_indices[np.arange(len(k_indices)) != k, :].flatten()
    
    # x data: train/test
    x_tr = tx[idx_tr, :]
    x_te = tx[idx_te, :]
    
    # y data: train/test
    y_tr = y[idx_tr]
    y_te = y[idx_te]
  
    # training ridge regression
    weights, _ = model(y_tr, x_tr, lambda_, **kw_model)

    # computing losses
    loss_tr = loss(y_tr, x_tr, weights, lambda_, **kw_loss)
    loss_te = loss(y_te, x_te, weights, lambda_, **kw_loss)
    
    return loss_tr, loss_te

def cross_validation_select(x, y, model, loss, kw_model = {}, kw_loss = {}, seed = 1, k_fold = 5, do_plot = True, do_tqdm = False, lambdas = None):
    if type(lambdas) == type(None):
        lambdas = np.logspace(-6, 1, 10)
    
    # split data in k fold
    k_indices = build_k_indices(len(y), k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr, rmse_te = [], []
    rmse = [rmse_tr, rmse_te]
    rmse_all = [[], []]
    
    for lambda_ in tqdm(lambdas) if do_tqdm else lambdas:
        rmse_ = [[], []]
        for k in range(k_fold):
            [rmse_[i].append(x) for i, x in
             enumerate(cross_validation(y, x, k_indices, k, model, kw_model, loss, kw_loss, lambda_))]
        [rmse[i].append(np.mean(x)) for (i, x) in enumerate(rmse_)]
        [rmse_all[i].append(x) for (i, x) in enumerate(rmse_)]
    
    rmse_all = np.array(rmse_all)
    idx_min = np.argmin(rmse_te)

    if do_plot:
        plots.cross_validation_visualization(lambdas, rmse_tr, rmse_te,
                                     all_data = [rmse_all[0].T, rmse_all[1].T])
    
    return idx_min, rmse_all, lambdas
