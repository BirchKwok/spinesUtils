import numpy as np
from scipy import stats


def reg_wb(x, y):
    """x, y都得为一维"""
    x_mean, y_mean = np.mean(x), np.mean(y)

    w = np.dot((x - x_mean), (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - w * x_mean

    return w, b, stats.trim_mean(y - (w * x + b), 0.1)  # weight, bias, residual
