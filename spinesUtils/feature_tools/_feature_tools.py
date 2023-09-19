"""特征工具集合"""
import os.path
from copy import deepcopy
from operator import gt, ge
from itertools import permutations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from spinesUtils.asserts import TypeAssert
from spinesUtils.utils import Printer
from spinesUtils.metrics import make_metric


@TypeAssert({'df': DataFrame, 'threshold': float})
def variation_threshold(df, threshold=0.01):
    """使用异众比例筛选特征"""
    from spinesUtils.data_insight import df_preview
    filters = df_preview(df, indicators=['variation']).to_dict()['variation']

    cols = []
    for col, v in filters.items():
        if v <= threshold:
            cols.append(col)

    return cols


@TypeAssert({'df': DataFrame, 'threshold': float})
def vars_threshold(df, threshold=0.0):
    """方差筛选法"""
    df_var = {}
    for i in select_numeric_cols(df):
        df_var[i] = df[i].var()

    cols = []
    for col, v in df_var.items():
        if v <= threshold:
            cols.append(col)

    return cols


class TreeSequentialFeatureSelector:
    @TypeAssert({
        'metrics_name': str,
        'cv': int,
        'cv_shuffle': bool,
        'forward': bool,
        'floating': bool,
        'random_state': int,
        'init_nums': (None, int),
        'baseline': (None, int),
        'fim': str,
        'verbose': (bool, int)
    }, func_name='FeatureSelector')
    def __init__(
            self,
            estimator,
            metrics_name='f1',
            cv=5,
            cv_shuffle=True,
            forward=True,
            floating=False,
            random_state=0,
            init_nums=None,
            baseline=None,
            fim='shap',
            early_stopping_rounds=None,
            log_file_path='feature_selection.log',
            best_features_save_path='best_feature.txt',
            verbose=True
    ):
        self.estimator = estimator
        self.cv = cv
        self.cv_shuffle = cv_shuffle
        self.cv_random_state = random_state

        self.init_nums = init_nums
        self.baseline = baseline
        self.fim = fim
        self.early_stopping_rounds = early_stopping_rounds

        self.score_func = make_metric(metrics_name)
        self.metrics_name = metrics_name
        self.forward = forward
        self.floating = floating

        self.logger = Printer(fp=log_file_path, verbose=verbose, truncate_file=True)

        self.log_file_path = log_file_path
        self.best_features_save_path = best_features_save_path

        self.verbose = verbose

        self.best_cols_ = None
        self.best_score_ = np.finfo(np.float32).min
        self.fitted = False

    @staticmethod
    def _zoom_list(list_a, list_b, append=True):
        """如果是顺序就是向列表中增加未存在的值，如果是逆序则向列表中删除已存在的值"""
        _ = deepcopy(list_a)
        if append:
            for i in list_b:
                if i not in _:
                    _.append(i)
        else:
            for i in list_b:
                if i in _:
                    _.remove(i)
        return _

    @staticmethod
    def _check_eval_set(eval_set):
        assert all([isinstance(i, tuple) for i in eval_set]) and isinstance(eval_set, list)

    def _batch_evaluate(self, est, x_cols, eval_set):
        """批量预估"""
        scores = []
        for s in eval_set:
            scores.append(self.score_func(s[1], est.predict(s[0][x_cols])))

        return np.round(np.mean(scores), 6)

    def _search_rollback(self, x, y, eval_set, current_searching_epoch, float_best_cols, rollbacks,
                         best_score):
        """回测缩减特征, 如果对整体性能有提升就加回来"""
        compare_f = gt  # 只有在对整体性能有提升的前提下才加进来回测特征

        # 尝试所有rollbacks组合
        for ml in range(1, len(rollbacks)):
            sorted_perms = [sorted(i) for i in permutations(rollbacks, ml)]
            perms = []

            for i in sorted_perms:
                if i not in perms:
                    perms.append(i)
            del sorted_perms

            for idx, _c in enumerate(perms):
                _test_cols = self._zoom_list(float_best_cols, _c, append=True)

                # 如果相等则已训练过
                if len(_test_cols) == self._zoom_list(float_best_cols, rollbacks, append=True):
                    continue

                est = deepcopy(self.estimator)
                est.fit(x[_test_cols], y)

                current_epoch_score = self._batch_evaluate(est, _test_cols, eval_set)

                if compare_f(current_epoch_score, best_score):
                    self.logger.save_log_and_throwout(
                        string=f"[float {current_searching_epoch}-{idx + 1}/{len(perms)}] - "
                               f"feas_num {len(_test_cols)}] # best loop,"
                               f"{self.metrics_name} score is: {current_epoch_score}")

                    float_best_cols = self._zoom_list(float_best_cols, _test_cols, append=True)
                    best_score = current_epoch_score

        return float_best_cols, best_score

    def _n_fold_dataset(self, x, y):
        from sklearn.model_selection import KFold

        k_fold = KFold(n_splits=self.cv, random_state=self.cv_random_state, shuffle=self.cv_shuffle)

        for i, (train_idx, test_idx) in enumerate(k_fold.split(x, y)):
            yield (x.iloc[train_idx, :].reset_index(drop=True), x.iloc[test_idx, :].reset_index(drop=True),
                   y[train_idx].reset_index(drop=True), y[test_idx].reset_index(drop=True))

    @TypeAssert({'x': pd.DataFrame, 'y': pd.Series, 'early_stopping_rounds': (int, None), 'patience': int})
    def fit(self, x, y, early_stopping_rounds=10, patience=10):
        self.logger.save_log_and_throwout(f"metrics: {self.metrics_name}")
        # 先根据异众比例剔除
        variation_chosen_cols = variation_threshold(x, 0.01)
        self.logger.save_log_and_throwout(f"excluded columns using the variation ratio: {variation_chosen_cols}")
        # 再剔除零方差的样本
        low_var_cols = vars_threshold(x, 0.0)
        self.logger.save_log_and_throwout(f"using zero variance: {low_var_cols}")

        to_delete_cols = set(variation_chosen_cols) | set(low_var_cols)

        cv_cols_set = set()

        if self.cv > 0:
            for x_train, x_test, y_train, y_test in self._n_fold_dataset(x.drop(columns=list(to_delete_cols)), y):
                cols, score = self._fit(x_train, y_train, eval_set=[(x_test, y_test)],
                                        early_stopping_rounds=early_stopping_rounds, patience=patience)
                for c in cols:
                    cv_cols_set.add(c)
            # evaluate on the whole dataset
            _chosen_cols = list(cv_cols_set)
            best_cols, best_score = self._fit(
                x[_chosen_cols], y, eval_set=[(x[_chosen_cols], y)],
                early_stopping_rounds=early_stopping_rounds, patience=patience
            )
        else:
            best_cols, best_score = self._fit(x, y, eval_set=[(x, y)],
                                              early_stopping_rounds=early_stopping_rounds, patience=patience)
        self.best_cols_ = best_cols
        self.best_score_ = best_score

        self.fitted = True
        return self

    @TypeAssert({
        'x': pd.DataFrame,
        'y': pd.Series,
        'eval_set': list,
        'early_stopping_rounds': (int, None),
        'patience': int
    })
    def _fit(self, x, y, eval_set, early_stopping_rounds=10, patience=20, init_cols=None):
        assert patience > 1
        self._check_eval_set(eval_set)

        estimator = deepcopy(self.estimator)

        # 获取初始模型
        init_model = estimator.fit(x, y)
        self.logger.save_log_and_throwout(string="The initial full model was successfully obtained.")
        # 获取初始模型特征重要性
        init_feas = feature_importances(init_model, eval_x=x,
                                        reverse=True, accoding_to=self.fim, target=y)['features'].values.tolist()

        # 初始评估特征个数
        init_nums = self.init_nums or 1

        if init_cols is None:
            # 至少有一个特征
            best_cols = init_feas[:init_nums] if self.forward else init_feas
        else:
            best_cols = init_cols

        # iter features
        # 先删除重要性低的特征，或者先添加重要性高的特征
        iter_feas = init_feas[init_nums:] if self.forward else init_feas[init_nums::-1]

        if self.baseline is None:
            best_score = np.finfo(np.float32).min if self.forward else self._batch_evaluate(init_model, best_cols, eval_set)
        else:
            best_score = self.baseline

        # 当前轮次忍耐次数
        current_patience = 1
        mid_cols = deepcopy(best_cols)
        rollbacks = []

        # 轮次停止检查计数器
        stopping_rounds = 0

        for idx, fea in enumerate(iter_feas):
            # 如果当前轮次已经达到忍耐次数
            if current_patience > patience:
                if self.early_stopping_rounds:
                    # 轮次停止检查计数器 + 1
                    stopping_rounds += 1

                # 如果使用浮动召回，就召回特征
                if self.floating:
                    best_cols, _c_score = \
                        self._search_rollback(x, y, eval_set, idx, best_cols,
                                              rollbacks, best_score)
                    if _c_score > best_score:
                        best_score = _c_score
                        # 当浮动召回有效，就重置轮次停止检查计数器
                        stopping_rounds = 0

                mid_cols = best_cols
                current_patience = 1
                rollbacks = []

            # 如果轮次停止检查计数器 ≥ early_stopping_rounds，跳出循环
            if early_stopping_rounds is not None and stopping_rounds >= early_stopping_rounds:
                break

            _test_cols = self._zoom_list(mid_cols, [fea], append=self.forward)  # forward append, backward remove

            # fit testing model
            estimator = deepcopy(self.estimator)
            estimator.fit(x[_test_cols], y)
            # Check if the model performance has improved
            current_loop_score = self._batch_evaluate(estimator, _test_cols, eval_set)

            loop_desc = f"[{idx + 1}/{len(iter_feas)}]- feas_num {len(_test_cols)} - "

            compare_func = gt if self.forward else ge  # > or >=

            if compare_func(current_loop_score, best_score):
                loop_desc += "# best loop, "
                # 如果浮动筛选
                if self.floating:
                    if len(rollbacks) == 0:
                        best_cols = self._zoom_list(best_cols, _test_cols, append=self.forward) \
                            if self.forward else self._zoom_list(best_cols, [fea], append=self.forward)
                    else:
                        if self.forward:
                            rollbacks = self._zoom_list(rollbacks, [fea], append=True)

                        best_cols, current_loop_score = self._search_rollback(
                            x, y,
                            eval_set, idx + 1,
                            best_cols,
                            rollbacks, current_loop_score
                        )
                else:
                    best_cols = self._zoom_list(best_cols, _test_cols, append=self.forward) \
                        if self.forward else self._zoom_list(best_cols, [fea], append=self.forward)

                rollbacks = []
                mid_cols = best_cols
                best_score = current_loop_score
                current_patience = 1
                # 保存最佳轮次特征
            else:
                if current_patience <= patience:
                    loop_desc += f"patience: {current_patience}, "
                    rollbacks = self._zoom_list(rollbacks, [fea], append=True)
                    mid_cols = self._zoom_list(mid_cols, _test_cols, append=self.forward) \
                        if self.forward else self._zoom_list(mid_cols, [fea], append=self.forward)
                    current_patience += 1

            loop_desc += (f" {self.metrics_name} score is: {current_loop_score}, "
                          f"current part of dataset best score is: {best_score}")

            self.logger.save_log_and_throwout(string=loop_desc)

        return best_cols, best_score

    def transform(self, x):
        assert self.fitted is True

        return x[self.best_cols_]


def feature_importances(model, eval_x=None, p_threshold=None, reverse=True, accoding_to='model', target=None):
    """输出模型重要性的百分比排名， 默认倒序排列
    model: 模型，要求必须具备feature_importances_属性
    p_threshold: 过滤阈值(百分比)，返回过滤后的阈值的特征，要求大于等于0，小于等于100, 或者None
    reverse: bool，是否翻转阈值结果，如果为True, 则返回大于等于过滤阈值的特征，否则返回小于等于过滤阈值的特征
    according_to: shap or model
    """
    assert all([hasattr(model, 'feature_importances_'), hasattr(model, 'feature_name_')])
    assert p_threshold is None or 0 <= p_threshold <= 100
    assert len(model.feature_name_) == len(model.feature_importances_)
    assert accoding_to in ('model', 'shap')

    if accoding_to == 'shap' and eval_x is None:
        raise ValueError("according_to参数等于shap时，eval_x不能为None")

    from ..metrics import sorted_shap_val

    if accoding_to == 'shap':
        importances = sorted_shap_val(model, eval_x, target=target)
    else:
        importances = Series(model.feature_importances_, index=model.feature_name_)

    importance_rate = importances / importances.sum()

    if p_threshold is not None:
        compare_with = ">=" if reverse else "<="

        return DataFrame([
            (i[0], i[1][0], f"{round(i[1][1] * 100, 2)}%") for i in
            filter(
                lambda s:
                eval(f"s[1][0] * 100 {compare_with} p_threshold", {'s': s, 'p_threshold': p_threshold}), sorted(
                    {
                        name: [value1, value2] for name, value1, value2 in
                        zip(importances.index, importances.values, importance_rate.values)
                    }.items(), key=lambda s: s[1], reverse=reverse
                )
            )
        ], columns=['features', 'importance_val', 'importance_rate'])

    return DataFrame([
        (i[0], i[1][0], f"{round(i[1][1] * 100, 2)}%") for i in
        sorted(
            {
                name: [value1, value2] for name, value1, value2 in
                zip(importances.index, importances.values, importance_rate.values)
            }.items(), key=lambda s: s[1], reverse=reverse
        )
    ], columns=['features', 'importance_val', 'importance_rate'])


def select_numeric_cols(dataset, exclude_binary_value_column=False) -> np.ndarray:
    import pandas as pd

    assert isinstance(dataset, pd.DataFrame)
    if exclude_binary_value_column:
        _ = dataset.nunique(axis=0)
        return _[_ > 2].index
    return dataset._get_numeric_data().columns


def get_specified_type_cols(df, types):
    """
    从pandas dataframe中快速选取对应类型数据
    """
    _ = []
    for k, v in df.dtypes.items():
        if isinstance(types, list):
            if v in types:
                _.append(k)
        elif isinstance(types, str):
            if v == types:
                _.append(k)
        else:
            raise ValueError("`types` only accept list or string type")
    return _


def get_x_cols(df, y_col, exclude_columns=None):
    assert isinstance(y_col, (str, list))
    assert exclude_columns is None or isinstance(exclude_columns, (str, list))
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    cols = df.columns.tolist()
    if isinstance(y_col, str):
        if y_col in cols:
            cols.remove(y_col)
    else:
        for yi in y_col:
            if yi in cols:
                cols.remove(yi)

    if exclude_columns is not None:
        if isinstance(exclude_columns, str):
            if exclude_columns in cols:
                cols.remove(exclude_columns)
        else:
            for ei in exclude_columns:
                if ei in cols:
                    cols.remove(ei)

    return cols


def exclude_columns(df, cols):
    """返回排除掉指定列的dataframe
    :params:

    df: pandas.core.DataFrame
    exclude_columns: 需要排除的列，可以是list、tuple、np.ndarray、pd.Series

    :returns:
    pd.core.DataFrame

    """
    import pandas as pd
    import numpy as np

    assert isinstance(df, pd.DataFrame)
    assert isinstance(cols, (list, tuple, np.ndarray, pd.Series))
    column_num = df.shape[1]

    if len(cols) / column_num < 0.3:
        return df.drop(columns=cols)

    return df.loc[:, ~df.columns.isin(cols)]
