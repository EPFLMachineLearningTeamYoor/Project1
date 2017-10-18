from matplotlib import pyplot as plt
import numpy as np
from mpl_scatter_density import ScatterDensityArtist
from tqdm import tqdm

def savefig(name):
    kwargs = {'bbox_inches': 'tight', 'transparent': False, 'pad_inches': 0}
    plt.savefig('./analysis/%s.pdf' % name, **kwargs)
    plt.savefig('./analysis/%s.png' % name, **kwargs)

def do_hist_scatter(X, y_, header, idx_x = None, idx_y = None, bins = 20):
    N = X.shape[1]
    if idx_x == None:
        idx_x = range(N)
    if idx_y == None:
        idx_y = range(N)
        
    _, plots = plt.subplots(len(idx_x), len(idx_y), figsize=(15,15))
    with tqdm(total=len(idx_x) * len(idx_y)) as pbar:
        for x, i in enumerate(idx_x):
            for y, j in enumerate(idx_y):
                ax = plots[x][y]
                ax.set_rasterized(True)
                if i == j:
                    ax.hist(X[:,i], color='green', bins=bins,
                            histtype='bar', align='mid')
                else:
                    ax.scatter(X[:,j], X[:,i],
                               c = ['red' if t > 0 else 'blue' for t in y_],
                               s=1)
                if x == 0 or x == len(idx_x) - 1:
                    ax.xaxis.set_label_position('top' if x == 0 else 'bottom')
                    ax.set_xlabel(header[j])
                if y == 0 or y == len(idx_y) - 1:
                    ax.yaxis.set_label_position('left' if y == 0 else 'right')
                    ax.set_ylabel(header[i])
                pbar.update(1)

def get_ranges(N, k):
    all_range = list(range(N))
    n_intervals = N // k
    if n_intervals * k < N: n_intervals += 1
    for i in range(n_intervals):
        for j in range(n_intervals):
            range_x = range(i * k, min(N, i * k + k))
            range_y = range(j * k, min(N, j * k + k))
            yield (range_x, range_y)

def do_hist(X, header, shape = (None, None), bins = 20):
    N = X.shape[1]
    assert(shape[0] * shape[1] >= N)
    _, plots = plt.subplots(shape[0], shape[1], figsize=(15,15))
    with tqdm(total=N) as pbar:
        for i in range(shape[0]):
            for j in range(shape[1]):
                n = i * shape[1] + j
                if n >= N:
                    continue
                ax = plots[i][j]
                ax.hist(X[:,n], color='green', bins=bins, histtype='bar', align='mid')
                ax.set_title(label = header[n], fontsize=7)
                ax.get_yaxis().set_tick_params(which='both', direction='in')
                ax.get_xaxis().set_tick_params(which='both', direction='in')
                pbar.update(1)

def do_Xy(X, y, header, shape = (None, None), bins = 15):
    N = X.shape[1]
    assert(shape[0] * shape[1] >= N)
    _, plots = plt.subplots(shape[0], shape[1], figsize=(15,15))
    with tqdm(total=N) as pbar:
        for i in range(shape[0]):
            for j in range(shape[1]):
                n = i * shape[1] + j
                if n >= N:
                    continue
                ax = plots[i][j]
                ax.hist(X[y > 0,n], bins=bins, alpha=0.5, label='+1')
                ax.hist(X[y <= 0,n], bins=bins, alpha=0.5, label='-1')
                ax.get_yaxis().set_tick_params(which='both', direction='in')
                ax.get_xaxis().set_tick_params(which='both', direction='in')
                ax.set_title(label = header[n], fontsize=7)
                if i == 0 and j == 0:
                    ax.legend()
                pbar.update(1)

def do_boxplot(X, header, shape = (None, None)):
    N = X.shape[1]
    assert(shape[0] * shape[1] >= N)
    _, plots = plt.subplots(shape[0], shape[1], figsize=(15,30))
    with tqdm(total=N) as pbar:
        for i in range(shape[0]):
            for j in range(shape[1]):
                n = i * shape[1] + j
                if n >= N:
                    continue
                ax = plots[i][j]
                ax.boxplot(X[:, n], vert = False)
                ax.get_yaxis().set_tick_params(which='both', direction='in')
                ax.get_xaxis().set_tick_params(which='both', direction='in')
                ax.set_title(label = header[n], fontsize=7)
                pbar.update(1)

