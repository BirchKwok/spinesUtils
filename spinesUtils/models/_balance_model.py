import datetime

import pandas as pd
import numpy as np


class ThresholdVotingClassifier:
    """
    平衡投票分类器, 适用于二分类
    """

    def __init__(self, meta_estimators) -> None:
        """多数类label需设置为0，少数类label为1"""
        assert isinstance(meta_estimators, list) and len(meta_estimators) == 2, \
            'meta_estimators 参数需要为list, 且长度为2.'
        self.myprint = self._myprint

        # model_L1 输出始终为负类(多数类)
        self.model_L1 = lambda arr: np.zeros(arr.shape[0])
        self.model_L2 = meta_estimators[0]
        self.model_L3 = meta_estimators[1]
        self._ests = meta_estimators

        self._classes = 2
        self._auto_threshold = {'Model L2': 0, 'Model L3': 0}

    @staticmethod
    def _myprint(s, silent=False) -> None:
        if not silent:
            print(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), s
            )

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

    def fit(
            self,
            X,
            y,
            threshold_search_set=None,
            second_model_input_size='auto',
            silent=False,
            params_dict=None):
        assert threshold_search_set is None or len(threshold_search_set) == 2
        assert isinstance(X, (np.ndarray, pd.DataFrame)) and isinstance(y, (np.ndarray, pd.Series))
        if isinstance(X, pd.DataFrame):
            cate_cols = ', '.join(X.select_dtypes('category').columns.tolist())
        else:
            cate_cols = ''

        type_x = 'pandas dataframe' if isinstance(X, pd.DataFrame) else 'numpy ndarray'
        type_y = 'pandas series' if isinstance(y, pd.Series) else 'numpy ndarray'

        self.myprint(f"[Main step] X type is {type_x}, y type is {type_y}.", silent=silent)

        self.myprint(f"[Category columns] {cate_cols}", silent=silent)
        self._classes = len(np.unique(y))

        assert self._classes == 2, f"Found {self._classes} classes from dataset."

        self.myprint("[Main step] Prepare to fit.", silent=silent)

        self.myprint(
            f"[Model L1] Training samples shape is {X.shape}, "
            f"sample size ratio is {self._count_ratio_of_sample_size(y)}"
        )

        self.myprint("[Main step] Model L1 fitted.", silent=silent)

        X_train_L2, y_train_L2 = self.split_balance_datasets_L2(
            X, y, second_model_input_size=second_model_input_size
        )
        self.myprint("[Main step] Datasets split for model L2.", silent=silent)
        self.myprint(
            f"[Model L2] Training samples shape is {X_train_L2.shape}, "
            f"sample size ratio is {self._count_ratio_of_sample_size(y_train_L2)}",
            silent=silent
        )

        if params_dict is not None and params_dict.get('L2', None) is not None:
            self.model_L2 = self.model_L2.fit(X_train_L2, y_train_L2, **params_dict['L2'])
        else:
            self.model_L2 = self.model_L2.fit(X_train_L2, y_train_L2)

        self.myprint("[Main step] Model L2 fitted.", silent=silent)

        X_train_L3, y_train_L3 = self.split_balance_datasets_L3(X, y)
        self.myprint("[Main step] Datasets split for model L3.", silent=silent)
        self.myprint(
            f"[Model L3] Training samples shape is {X_train_L3.shape}, "
            f"sample size ratio is {self._count_ratio_of_sample_size(y_train_L3)}",
            silent=silent
        )
        if params_dict is not None and params_dict.get('L3', None) is not None:
            self.model_L3 = self.model_L3.fit(X_train_L3, y_train_L3, **params_dict['L3'])
        else:
            self.model_L3 = self.model_L3.fit(X_train_L3, y_train_L3)

        self.myprint("[Main step] Model L3 fitted.", silent=silent)

        if threshold_search_set:
            self.myprint(f"[Main step] start to search probability threshold...", silent=silent)
            self._auto_search_threshold(self.model_L2, threshold_search_set,
                                        'Model L2', False)
            self._auto_search_threshold(self.model_L3, threshold_search_set,
                                        'Model L3', False)

        return self

    def _auto_search_threshold(self, model, threshold_search_set, model_name, silent=False):
        """
        threshold_search_set: 需要在上面寻找最佳概率划分阈值的测试集，形如[X_test, y_test]
        """
        from sklearn.metrics import f1_score
        max_f1 = 0

        init_yp = model.predict_proba(threshold_search_set[0])[:, -1].squeeze()

        for ts in np.arange(0, 1, 0.01):
            yp = self._threshold_chosen(
                init_yp,
                threshold=ts
            )

            f1_epoch = f1_score(threshold_search_set[1], yp)
            if f1_epoch > max_f1:
                self.myprint(f"[{model_name} searching] threshold:{ts}, f1_score:{f1_epoch}.", silent=silent)
                self._auto_threshold[model_name] = ts
                max_f1 = f1_epoch

        self.myprint(
            f"[{model_name} best threshold] threshold:{self._auto_threshold[model_name]}, "
            f"f1_score:{max_f1}.",
            silent=silent
        )

        self.myprint(
            f"[Main step] {model_name} threshold auto set to {self._auto_threshold[model_name]}.",
            silent=silent
        )

    @staticmethod
    def _threshold_chosen(y, threshold=0.2):
        """二分类阈值选择法"""
        return (y > threshold).astype(int)

    def _predict(self, X, method='soft', silent=False):
        self.myprint(f"[Main step] Prepare to predict. Predict mode:{method}")
        if method == 'soft':
            assert any([hasattr(i, 'predict_proba') for i in self._ests]), \
                "Estimators must have `predict_proba` attribute."
            self.myprint("[Main step] Model L1 has been predicted.", silent=silent)
            y_pred_L2 = self.model_L2.predict_proba(X)[:, 1].reshape((-1, 1))
            self.myprint("[Main step] Model L2 has been predicted.", silent=silent)
            y_pred_L3 = self.model_L3.predict_proba(X)[:, 1].reshape((-1, 1))
            self.myprint("[Main step] Model L3 has been predicted.", silent=silent)
        else:
            self.myprint("[Main step] Model L1 has been predicted.", silent=silent)
            y_pred_L2 = self.model_L2.predict(X).reshape((-1, 1))
            self.myprint("[Main step] Model L2 has been predicted.", silent=silent)
            y_pred_L3 = self.model_L3.predict(X).reshape((-1, 1))
            self.myprint("[Main step] Model L3 has been predicted.", silent=silent)

        return y_pred_L2, y_pred_L3

    def predict(self, X, vote_method='soft', silent=False):
        assert isinstance(vote_method, str) and vote_method in ('soft', 'hard')
        if vote_method == 'hard':
            y_pred_L2, y_pred_L3 = self._predict(X, 'hard', silent=silent)
        else:
            y_pred_L2, y_pred_L3 = self._predict(X, 'soft', silent=silent)
            y_pred_L2 = self._threshold_chosen(y_pred_L2, threshold=self._auto_threshold['Model L2'])
            y_pred_L3 = self._threshold_chosen(y_pred_L3, threshold=self._auto_threshold['Model L3'])

        yps = np.concatenate((y_pred_L2, y_pred_L3), axis=1).sum(axis=1).squeeze()

        return np.where(yps == 2, 1, 0)

    def predict_proba(self, X, silent=False):
        y_pred_L2, y_pred_L3 = self._predict(X, 'soft', silent=silent)

        yps = np.concatenate((y_pred_L2, y_pred_L3), axis=1).sum(axis=1).squeeze()
        return np.where(yps == 2, np.max([y_pred_L2, y_pred_L3], axis=0), np.min([y_pred_L2, y_pred_L3], axis=0))

