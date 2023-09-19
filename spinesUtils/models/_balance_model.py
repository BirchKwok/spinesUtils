import pandas as pd
import numpy as np
from frozendict import frozendict

from spinesUtils.asserts import TypeAssert
from spinesUtils.utils import Printer


class BinaryBalanceClassifier:
    """
    Balanced voting classifier with threshold, suitable for binary classification
    """
    @TypeAssert({'meta_estimators': (list, tuple), 'verbose': bool}, 
                func_name='ThresholdVotingClassifier')
    def __init__(self, meta_estimators, verbose=True) -> None:
        """多数类label需设置为0，少数类label为1"""
        assert len(meta_estimators) == 2, \
            'The length of meta_estimators must be equal to 2.'

        # model_L1 输出始终为负类(多数类)
        self.model_L1 = lambda arr: np.zeros(arr.shape[0])
        self.model_L2 = meta_estimators[0]
        self.model_L3 = meta_estimators[1]
        self._ests = meta_estimators

        self._classes = 2
        self.auto_threshold = {'Model L2': 0, 'Model L3': 0}
        
        self.logger = Printer(verbose=verbose)

    @TypeAssert({
        'X_train': (np.ndarray, pd.DataFrame),
        'y_train': (np.ndarray, pd.Series),
        'second_model_input_size': str
    })
    def split_balance_datasets_L2(self, X_train, y_train, second_model_input_size='auto'):
        """返回model-L2的X_train和y_train"""

        y_pred = self.model_L1(X_train)

        if isinstance(X_train, np.ndarray):
            right_set_x = X_train[y_train == y_pred]
            wrong_set_x = X_train[y_train != y_pred]
            concat_x = np.concatenate
        else:
            right_set_x = X_train.iloc[np.argwhere(np.asarray(y_train == y_pred)).squeeze(), :].reset_index(drop=True)
            wrong_set_x = X_train.iloc[np.argwhere(np.asarray(y_train != y_pred)).squeeze(), :].reset_index(drop=True)
            concat_x = pd.concat

        if second_model_input_size == 'auto':
            second_model_input_size = min([len(right_set_x), len(wrong_set_x)])

        if isinstance(y_train, pd.Series):
            right_set_y = y_train.iloc[np.argwhere(np.asarray(y_train == y_pred)).squeeze()].reset_index(drop=True)
            wrong_set_y = y_train.iloc[np.argwhere(np.asarray(y_train != y_pred)).squeeze()].reset_index(drop=True)
            concat_y = pd.concat
        else:
            right_set_y = y_train[y_train == y_pred]
            wrong_set_y = y_train[y_train != y_pred]
            concat_y = np.concatenate

        x = concat_x(
            (right_set_x[:second_model_input_size], wrong_set_x[:second_model_input_size]), axis=0
        )

        y = concat_y(
            (right_set_y[:second_model_input_size], wrong_set_y[:second_model_input_size]), axis=0
        )

        return x, y

    @TypeAssert({
        'X_train': (np.ndarray, pd.DataFrame),
        'y_train': (np.ndarray, pd.Series)
    })
    def split_balance_datasets_L3(self, X_train, y_train):
        """返回model-L3的X_train和y_train"""
        y_pred_1 = self.model_L1(X_train)
        y_pred_2 = self.model_L2.predict(X_train).squeeze()

        if isinstance(X_train, np.ndarray):
            x = X_train[np.asarray(y_pred_1 != y_pred_2).squeeze()]
        else:

            x = X_train.iloc[np.argwhere(np.asarray(y_pred_1 != y_pred_2)).squeeze(), :].reset_index(drop=True)

        if isinstance(y_train, np.ndarray):
            y = y_train[y_pred_1 != y_pred_2]
        else:
            y = y_train.iloc[np.argwhere(np.asarray(y_pred_1 != y_pred_2)).squeeze()].reset_index(drop=True)

        return x, y

    @staticmethod
    def _count_ratio_of_sample_size(y) -> dict:
        from collections import Counter

        c = Counter(y)
        sum_c = sum(c.values())
        _ = {i: round(c[i] / sum_c, 2) for i in c.keys()}
        return dict(sorted(_.items(), key=lambda s: s[0]))

    @TypeAssert({
        'X': (np.ndarray, pd.DataFrame),
        'y': (np.ndarray, pd.Series),
        'threshold_search_set': (None, list, tuple)
    })
    def fit(
            self,
            X,
            y,
            threshold_search_set=None,
            second_model_input_size='auto',
            params_dict=None):
        if isinstance(X, pd.DataFrame):
            cate_cols = ', '.join(X.select_dtypes('category').columns.tolist())
        else:
            cate_cols = ''

        type_x = 'pandas dataframe' if isinstance(X, pd.DataFrame) else 'numpy ndarray'
        type_y = 'pandas series' if isinstance(y, pd.Series) else 'numpy ndarray'

        self.logger.print(f"[Main step] X type is {type_x}, y type is {type_y}.")

        self.logger.print(f"[Category columns] {cate_cols}")
        self._classes = len(np.unique(y))

        assert self._classes == 2, f"Found {self._classes} classes from dataset."

        self.logger.print("[Main step] Prepare to fit.")

        self.logger.print(
            f"[Model L1] Training samples shape is {X.shape}, "
            f"sample size ratio is {self._count_ratio_of_sample_size(y)}"
        )

        self.logger.print("[Main step] Model L1 fitted.")

        X_train_L2, y_train_L2 = self.split_balance_datasets_L2(
            X, y, second_model_input_size=second_model_input_size
        )
        self.logger.print("[Main step] Datasets split for model L2.")
        self.logger.print(
            f"[Model L2] Training samples shape is {X_train_L2.shape}, "
            f"sample size ratio is {self._count_ratio_of_sample_size(y_train_L2)}"
        )

        if params_dict is not None and params_dict.get('L2', None) is not None:
            self.model_L2 = self.model_L2.fit(X_train_L2, y_train_L2, **params_dict['L2'])
        else:
            self.model_L2 = self.model_L2.fit(X_train_L2, y_train_L2)

        self.logger.print("[Main step] Model L2 fitted.")

        X_train_L3, y_train_L3 = self.split_balance_datasets_L3(X, y)
        self.logger.print("[Main step] Datasets split for model L3.")
        self.logger.print(
            f"[Model L3] Training samples shape is {X_train_L3.shape}, "
            f"sample size ratio is {self._count_ratio_of_sample_size(y_train_L3)}"
        )
        if params_dict is not None and params_dict.get('L3', None) is not None:
            self.model_L3 = self.model_L3.fit(X_train_L3, y_train_L3, **params_dict['L3'])
        else:
            self.model_L3 = self.model_L3.fit(X_train_L3, y_train_L3)

        self.logger.print("[Main step] Model L3 fitted.")

        if threshold_search_set:
            self.logger.print(f"[Main step] start to search probability threshold...")
            self._auto_search_threshold(self.model_L2, threshold_search_set,
                                        'Model L2')
            self._auto_search_threshold(self.model_L3, threshold_search_set,
                                        'Model L3')

            self.auto_threshold = frozendict(**self.auto_threshold)

        return self

    def _auto_search_threshold(self, model, threshold_search_set, model_name):
        """
        threshold_search_set: 需要在上面寻找最佳概率划分阈值的测试集，形如[X_test, y_test]
        """
        from sklearn.metrics import f1_score
        max_f1 = 0
        from spinesUtils._thresholds import auto_search_threshold
        best_threshold = auto_search_threshold(threshold_search_set[0], threshold_search_set[1], model=model)

        self.auto_threshold[model_name] = best_threshold
        self.logger.print(
            f"[{model_name} best threshold] threshold:{self.auto_threshold[model_name]}, "
            f"f1_score:{max_f1}."
        )

        self.logger.print(
            f"[Main step] {model_name} threshold auto set to {self.auto_threshold[model_name]}."
        )

    @staticmethod
    def _threshold_chosen(y, threshold=0.2):
        """二分类阈值选择法"""
        return (y > threshold).astype(int)

    def _predict(self, X, method='soft'):
        self.logger.print(f"[Main step] Prepare to predict. Predict mode:{method}")
        if method == 'soft':
            assert any([hasattr(i, 'predict_proba') for i in self._ests]), \
                "Estimators must have `predict_proba` attribute."

            self.logger.print("[Main step] Model L1 has been predicted.")
            y_pred_L2 = self.model_L2.predict_proba(X)[:, 1].reshape((-1, 1))
            self.logger.print("[Main step] Model L2 has been predicted.")
            y_pred_L3 = self.model_L3.predict_proba(X)[:, 1].reshape((-1, 1))
            self.logger.print("[Main step] Model L3 has been predicted.")
        else:
            self.logger.print("[Main step] Model L1 has been predicted.")
            y_pred_L2 = self.model_L2.predict(X).reshape((-1, 1))
            self.logger.print("[Main step] Model L2 has been predicted.")
            y_pred_L3 = self.model_L3.predict(X).reshape((-1, 1))
            self.logger.print("[Main step] Model L3 has been predicted.")

        return y_pred_L2, y_pred_L3

    @TypeAssert({'X': (np.ndarray, pd.DataFrame), 'vote_method': str})
    def predict(self, X, vote_method='soft'):
        assert vote_method in ('soft', 'hard')
        if vote_method == 'hard':
            y_pred_L2, y_pred_L3 = self._predict(X, 'hard')
        else:
            y_pred_L2, y_pred_L3 = self._predict(X, 'soft')
            y_pred_L2 = self._threshold_chosen(y_pred_L2, threshold=self.auto_threshold['Model L2'])
            y_pred_L3 = self._threshold_chosen(y_pred_L3, threshold=self.auto_threshold['Model L3'])

        yps = np.concatenate((y_pred_L2, y_pred_L3), axis=1).sum(axis=1).squeeze()

        return np.where(yps == 2, 1, 0)

    @TypeAssert({'X': (np.ndarray, pd.DataFrame)})
    def predict_proba(self, X):
        y_pred_L2, y_pred_L3 = self._predict(X, 'soft')

        yps = np.concatenate((y_pred_L2, y_pred_L3), axis=1).sum(axis=1).squeeze()
        return np.where(yps == 2, np.max([y_pred_L2, y_pred_L3], axis=0), np.min([y_pred_L2, y_pred_L3], axis=0))
