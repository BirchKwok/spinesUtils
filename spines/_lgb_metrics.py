# 自定义lightgbm的评估函数
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np


def lgb_skapi_f1_score(y_true, y_hat):
    y_hat = np.round(y_hat)
    return 'f1', f1_score(y_true, y_hat), True


def lgb_skapi_recall_score(y_true, y_hat):
    y_hat = np.round(y_hat)
    return 'recall', recall_score(y_true, y_hat), True


def lgb_skapi_precision_score(y_true, y_hat):
    y_hat = np.round(y_hat)
    return 'precision', precision_score(y_true, y_hat), True
