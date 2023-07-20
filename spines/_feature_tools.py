"""特征工具集合"""
import os.path
from copy import deepcopy
from operator import gt, ge

import numpy as np
from tqdm.auto import tqdm


def variation_threshold(df, threshold=0.01):
    """使用异众比例筛选特征"""
    from ._df_preview import df_preview
    filters = df_preview(df, indicators=['variation']).to_dict()['variation']

    cols = []
    for col, v in filters.items():
        if v <= threshold:
            cols.append(col)

    return cols


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


class FeatureSelector:
    def __init__(
            self,
            estimator,
            metrics='f1',
            forward=True,
            floating=True,
            log_file_path='feature_selection.log',
            best_features_save_path='best_feature.txt'
    ):
        self.estimator = estimator

        self.score_func = self._make_metrics(metrics)
        self.metrics_name = metrics
        self.forward = forward
        self.floating = floating

        from functools import partial
        from ._utils import log2file
        self.log2file_partial = partial(log2file, fp=log_file_path, line_end='\n', access_way='a', throw_out=True)

        self.log_file_path = log_file_path
        self.best_features_save_path = best_features_save_path
        # 清空log文件
        self._truncate_file()

        self.cols = None
        self.fitted = False

    @staticmethod
    def _make_metrics(metrics):
        if metrics == 'f1':
            from sklearn.metrics import f1_score as score_func
        elif metrics == 'recall':
            from sklearn.metrics import recall_score as score_func
        elif metrics == 'precision':
            from sklearn.metrics import precision_score as score_func
        elif metrics == 'accuracy':
            from sklearn.metrics import accuracy_score as score_func
        elif metrics == 'mse':
            from sklearn.metrics import mean_squared_error as score_func
        elif metrics == 'mae':
            from sklearn.metrics import mean_absolute_error as score_func
        elif metrics == 'r2':
            from sklearn.metrics import r2_score as score_func
        else:
            raise ValueError(f"{metrics} is invalid.")

        return score_func

    def _truncate_file(self):
        if self.log_file_path is not None:
            if os.path.isfile(self.log_file_path):
                os.truncate(self.log_file_path, 0)

        if self.best_features_save_path is not None:
            if os.path.isfile(self.best_features_save_path):
                os.truncate(self.best_features_save_path, 0)

    def _check_feas_to_save(self, feas, score):
        if self.best_features_save_path is None:
            return

        with open(self.best_features_save_path, 'a') as f:
            f.write(f'score: {score}, features: {feas}' + '\n')
        return

    @staticmethod
    def _show_list(list_, show_length=4):
        if show_length <= 0:
            return list_
        if show_length == 1:
            return [list_[0]] + ['...']

        return list_[:show_length // 2] + ['...'] + list_[-(show_length // 2):]

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

    def batch_evaluate(self, est, x_cols, eval_set):
        """批量预估"""
        scores = []
        for s in eval_set:
            scores.append(self.score_func(s[1], est.predict(s[0][x_cols])))

        return np.round(np.mean(scores), 6)

    def _search_rollback(self, x, y, eval_set, current_searching_epoch, columns, rollbacks,
                         best_score, max_len, feas_show_length=15):
        """回测缩减特征, 如果对整体性能有提升就加回来"""
        from itertools import permutations

        compare_f = gt  # 只有在对整体性能有提升的前提下才加进来回测特征
        _loop_num = 0
        for ml in range(1, max_len):
            if ml == 1:
                if self.forward:
                    _test_cols = self._zoom_list(columns, [rollbacks[-1]], append=True)

                    est = deepcopy(self.estimator)
                    est.fit(x[_test_cols], y)

                    current_epoch_score = self.batch_evaluate(est, _test_cols, eval_set)

                    if compare_f(current_epoch_score, best_score):
                        self.log2file_partial(
                            string=f"[Searching loop "
                                   f"{current_searching_epoch}-1/** - "
                                   f"Floating: feas_num {len(_test_cols)}]# 当前为最佳轮次, "
                                   f"当前特征列为: {self._show_list(_test_cols, show_length=feas_show_length)}, "
                                   f"{self.metrics_name}分数为: {current_epoch_score}")
                        columns = self._zoom_list(columns, _test_cols, append=True)
                        best_score = current_epoch_score
            else:
                perms = [i for i in permutations(rollbacks, ml)]

                t = []
                _loop_num = 1
                for _c in tqdm(perms,
                               desc=f"Searching loop {current_searching_epoch} - Epoch {ml} Floating filtration ...",
                               total=len(perms)):
                    _c = list(_c)
                    _test_cols = self._zoom_list(columns, _c, append=True)

                    # 如果相等则已训练过
                    if len(_test_cols) == self._zoom_list(columns, rollbacks, append=True):
                        continue

                    # 已训练过的直接跳过
                    if set(_test_cols) in t:
                        continue

                    t.append(set(_test_cols))

                    est = deepcopy(self.estimator)
                    est.fit(x[_test_cols], y)

                    current_epoch_score = self.batch_evaluate(est, _test_cols, eval_set)

                    if compare_f(current_epoch_score, best_score):
                        self.log2file_partial(
                            string=f"[Searching loop {current_searching_epoch}-{_loop_num + 1}/{len(perms)} - "
                                   f"Floating: feas_num {len(_test_cols)}] # 当前为最佳轮次,"
                                   f"当前特征列为: {self._show_list(_test_cols, show_length=feas_show_length)}, "
                                   f"{self.metrics_name}分数为: {current_epoch_score}")
                        columns = self._zoom_list(columns, _test_cols, append=True)
                        best_score = current_epoch_score

                    _loop_num += 1

        return columns, best_score

    def fit(self, x, y, eval_set, init_nums=30, patience=10, floating=True, baseline=None, fim='shap',
            feas_show_length=15, early_stopping_patience=None):
        assert isinstance(early_stopping_patience, int) or early_stopping_patience is None

        config_string = f"metrics: {self.metrics_name}" + ', '

        self._check_eval_set(eval_set)

        # 先根据异众比例剔除
        variation_chosen_cols = variation_threshold(x, 0.01)
        config_string += f"异众比例剔除列: {variation_chosen_cols}, "
        # 再剔除零方差的样本
        low_var_cols = vars_threshold(x, 0.0)
        config_string += f"零方差剔除列: {low_var_cols}"

        self.log2file_partial(string=config_string)
        to_delete_cols = set(variation_chosen_cols) | set(low_var_cols)

        est = deepcopy(self.estimator)

        # 获取初始模型
        init_model = est.fit(x.drop(columns=list(to_delete_cols)), y)
        self.log2file_partial(string="成功获取初始全量模型")
        # 获取初始模型特征重要性
        init_feas = feature_importances(init_model, eval_x=x.drop(columns=list(to_delete_cols)),
                                        reverse=True, accoding_to=fim, target=y)['features'].values.tolist()

        # 初始评估特征集合
        init_nums = init_nums - 1
        self.cols = init_feas[:init_nums] if self.forward else init_feas
        iter_feas = init_feas[init_nums:] if self.forward else init_feas[init_nums::-1]
        best_score = 0 if self.forward else self.batch_evaluate(init_model, self.cols, eval_set)

        best_score = baseline if baseline is not None else best_score

        current_patience = 1
        mid_cols = deepcopy(self.cols)
        rollbacks = []

        is_forward = '顺序' if self.forward else '逆序'

        loop_idx = 1
        stopping_patience = 0

        # 默认从10个特征开始筛选
        for fea in tqdm(iter_feas, desc=f"开始{is_forward}特征筛选...", total=len(iter_feas)):
            if current_patience > patience:
                if early_stopping_patience:
                    stopping_patience += 1

                if floating:
                    self.cols, _c_score = \
                        self._search_rollback(x, y, eval_set, loop_idx - 1, self.cols,
                                              rollbacks, best_score, max_len=len(rollbacks),
                                              feas_show_length=feas_show_length)
                    if _c_score > best_score:
                        best_score = _c_score
                        # 保存最佳轮次特征
                        self._check_feas_to_save(self.cols, _c_score)
                        stopping_patience = 0

                mid_cols = self.cols
                current_patience = 1
                rollbacks = []

            if stopping_patience >= early_stopping_patience:
                break

            _test_cols = self._zoom_list(mid_cols, [fea], append=self.forward)  # forward append, backward remove
            est = deepcopy(self.estimator)
            est.fit(x[_test_cols], y)

            current_loop_score = self.batch_evaluate(est, _test_cols, eval_set)

            if early_stopping_patience:
                loop_desc = f"[Searching loop {loop_idx} - feas_num {len(_test_cols)} - esp {stopping_patience}] "
            else:
                loop_desc = f"[Searching loop {loop_idx} - feas_num {len(_test_cols)}] "

            compare_func = gt if self.forward else ge  # > or >=
            if compare_func(current_loop_score, best_score):
                loop_desc += "# 当前为最佳轮次, "
                # 如果浮动筛选
                if floating:
                    if len(rollbacks) == 0:
                        self.cols = self._zoom_list(self.cols, _test_cols, append=self.forward) \
                            if self.forward else self._zoom_list(self.cols, [fea], append=self.forward)
                    else:
                        if self.forward:
                            rollbacks = self._zoom_list(rollbacks, [fea], append=True)

                        self.cols, current_loop_score = self._search_rollback(x, y, eval_set, loop_idx, self.cols,
                                                                              rollbacks, current_loop_score,
                                                                              max_len=len(rollbacks),
                                                                              feas_show_length=feas_show_length)
                else:
                    self.cols = self._zoom_list(self.cols, _test_cols, append=self.forward) \
                        if self.forward else self._zoom_list(self.cols, [fea], append=self.forward)

                rollbacks = []
                mid_cols = self.cols
                best_score = current_loop_score
                current_patience = 1
                # 保存最佳轮次特征
                self._check_feas_to_save(self.cols, current_loop_score)
            else:
                if current_patience <= patience:
                    loop_desc += f"patience: {current_patience}, "
                    rollbacks = self._zoom_list(rollbacks, [fea], append=True)
                    mid_cols = self._zoom_list(mid_cols, _test_cols, append=self.forward) \
                        if self.forward else self._zoom_list(mid_cols, [fea], append=self.forward)
                    current_patience += 1

            loop_desc += f"当前特征列为: {self._show_list(_test_cols, show_length=feas_show_length)}, {self.metrics_name}分数为: {current_loop_score}, " \
                         f"最佳{self.metrics_name}分数为: {best_score}"

            self.log2file_partial(string=loop_desc)

            loop_idx += 1

        self.fitted = True
        return self

    def transform(self, x):
        assert self.fitted is True

        return x[self.cols]


def feature_importances(model, eval_x=None, p_threshold=None, reverse=True, accoding_to='tree', target=None):
    """输出模型重要性的百分比排名， 默认倒序排列
    model: 模型，要求必须具备feature_importances_属性
    p_threshold: 过滤阈值(百分比)，返回过滤后的阈值的特征，要求大于等于0，小于等于100, 或者None
    reverse: bool，是否翻转阈值结果，如果为True, 则返回大于等于过滤阈值的特征，否则返回小于等于过滤阈值的特征
    according_to: shap or tree
    """
    assert all([hasattr(model, 'feature_importances_'), hasattr(model, 'feature_name_')])
    assert p_threshold is None or 0 <= p_threshold <= 100
    assert len(model.feature_name_) == len(model.feature_importances_)
    assert accoding_to in ('tree', 'shap')

    if accoding_to == 'shap' and eval_x is None:
        raise ValueError("according_to参数等于shap时，eval_x不能为None")

    from pandas import DataFrame, Series
    from .metrics import sorted_shap_val

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
                # zip(model.feature_name_, model.feature_importances_,
                #     model.feature_importances_ / model.feature_importances_.sum())
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
