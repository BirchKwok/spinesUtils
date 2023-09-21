import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert


@ParameterTypeAssert({'df': DataFrame, 'threshold': float})
@ParameterValuesAssert({'threshold': 'lambda s: 0 <= s <= 1'})
def variation_threshold(df, threshold=0.01):
    """使用异众比例筛选特征"""
    from spinesUtils.data_insight import df_preview
    filters = df_preview(df, indicators=['variation']).to_dict()['variation']

    cols = []
    for col, v in filters.items():
        if v <= threshold:
            cols.append(col)

    return cols


@ParameterTypeAssert({'df': DataFrame, 'threshold': float})
@ParameterValuesAssert({'threshold': 'lambda s: 0 <= s <= 1'})
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


@ParameterTypeAssert({
    'eval_x': (None, pd.DataFrame, np.ndarray),
    'p_threshold': (None, int),
    'reverse': bool,
    'according_to': str,
    'target': (None, pd.Series, np.ndarray)
})
def feature_importances(model, eval_x=None, p_threshold=None, reverse=True, according_to='model', target=None):
    """输出模型重要性的百分比排名， 默认倒序排列
    model: 模型，要求必须具备feature_importances_属性
    p_threshold: 过滤阈值(百分比)，返回过滤后的阈值的特征，要求大于等于0，小于等于100, 或者None
    reverse: bool，是否翻转阈值结果，如果为True, 则返回大于等于过滤阈值的特征，否则返回小于等于过滤阈值的特征
    according_to: shap or model
    """
    assert all([hasattr(model, 'feature_importances_'), hasattr(model, 'feature_name_')])
    assert p_threshold is None or 0 <= p_threshold <= 100
    assert len(model.feature_name_) == len(model.feature_importances_)
    assert according_to in ('model', 'shap')

    if according_to == 'shap' and eval_x is None:
        raise ValueError("according_to参数等于shap时，eval_x不能为None")

    from ..metrics import sorted_shap_val

    if according_to == 'shap':
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


@ParameterTypeAssert({
    'df': pd.DataFrame,
    'exclude_binary_value_column': bool
})
def select_numeric_cols(df, exclude_binary_value_column=False):
    if exclude_binary_value_column:
        _ = df.nunique(axis=0)
        return _[_ > 2].index
    return df._get_numeric_data().columns


@ParameterTypeAssert({
    'df': pd.DataFrame,
    'types': (str, list)
})
def get_specified_type_cols(df, types):
    """
    从pandas dataframe中快速选取对应类型数据
    """
    _ = []
    for k, v in df.dtypes.items():
        if isinstance(types, list):
            if v in types:
                _.append(k)
        else:
            if v == types:
                _.append(k)
    return _


@ParameterTypeAssert({
    'df': pd.DataFrame,
    'y_col': (str, list),
    'exclude_cols': (None, str, list)
})
def get_x_cols(df, y_col, exclude_cols=None):
    cols = df.columns.tolist()
    if isinstance(y_col, str):
        if y_col in cols:
            cols.remove(y_col)
    else:
        for yi in y_col:
            if yi in cols:
                cols.remove(yi)

    if exclude_cols is not None:
        if isinstance(exclude_cols, str):
            if exclude_cols in cols:
                cols.remove(exclude_cols)
        else:
            for ei in exclude_cols:
                if ei in cols:
                    cols.remove(ei)

    return cols


@ParameterTypeAssert({
    'df': pd.DataFrame,
    'cols': (list, tuple, np.ndarray, pd.Series)
})
def exclude_columns(df, cols):
    """返回排除掉指定列的dataframe
    :params:

    df: pandas.core.DataFrame
    exclude_columns: 需要排除的列，可以是list、tuple、np.ndarray、pd.Series

    :returns:
    pd.core.DataFrame

    """
    column_num = df.shape[1]

    if len(cols) / column_num < 0.3:
        return df.drop(columns=cols)

    return df.loc[:, ~df.columns.isin(cols)]
