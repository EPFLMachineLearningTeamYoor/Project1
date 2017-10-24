# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te, all_data = None):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    plt.xscale("log")
    plt.plot(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.plot(lambds, mse_te, marker=".", color='r', label='test error')
#    if type(all_data) != type(None):
#        widths = (lambds - np.hstack([lambds[1:], [0.5]])) / 2.
#        plt.boxplot(all_data[0], positions = lambds, widths = widths, manage_xticks = False, showfliers = False)
#        plt.boxplot(all_data[1], positions = lambds, widths = widths, manage_xticks = False, showfliers = False)
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
