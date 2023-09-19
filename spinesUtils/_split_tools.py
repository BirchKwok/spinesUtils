import numpy as np
import pandas as pd

from spinesUtils.asserts import TypeAssert


@TypeAssert({
    'df': pd.DataFrame,
    'x_cols': (list, tuple, pd.Series, np.ndarray),
    'y_col': str,
    'shuffle': bool,
    'return_valid': bool,
    'random_state': (None, int),
    'train_size': float,
    'valid_size': float,
    'stratify': (list, tuple, pd.Series, np.ndarray, None),
    'reset_index': bool
})
def train_test_split_bigdata_df(
        df,
        x_cols,
        y_col,
        shuffle=True,
        return_valid=True,
        random_state=42,
        train_size=0.8,
        valid_size=0.5,
        stratify=None,
        reset_index=True
):
    """切割大型数据集
    
    默认返回三个数据集: 训练集、验证集、测试集, 数据集包含y_col

    训练集、验证集、测试集默认比例为 8:1:1
    """

    from sklearn.model_selection import train_test_split

    X_train_idx, test_xs_idx, _, test_ys = train_test_split(
        df.index.values,  # 使用索引切割
        df[y_col].values,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify
    )
    if not return_valid:
        if reset_index:
            return df.iloc[X_train_idx, :][[*x_cols, y_col]].reset_index(drop=True), \
                df.iloc[test_xs_idx, :][[*x_cols, y_col]].reset_index(drop=True)

        return df.iloc[X_train_idx, :][[*x_cols, y_col]], df.iloc[test_xs_idx, :][[*x_cols, y_col]]

    if stratify is not None:
        valid_test_stratify = test_ys
    else:
        valid_test_stratify = None

    X_valid_idx, X_test_idx, _, _ = train_test_split(
        test_xs_idx,
        test_ys,
        train_size=valid_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=valid_test_stratify
    )

    if reset_index:
        return df.iloc[X_train_idx, :][[*x_cols, y_col]].reset_index(drop=True), \
            df.iloc[X_valid_idx, :][[*x_cols, y_col]].reset_index(drop=True), \
            df.iloc[X_test_idx, :][[*x_cols, y_col]].reset_index(drop=True)

    return df.iloc[X_train_idx, :][[*x_cols, y_col]], df.iloc[X_valid_idx, :][[*x_cols, y_col]], \
        df.iloc[X_test_idx, :][[*x_cols, y_col]]


@TypeAssert({
    'df': pd.DataFrame,
    'x_cols': (list, tuple, pd.Series, np.ndarray),
    'y_col': str,
    'shuffle': bool,
    'return_valid': bool,
    'random_state': (None, int),
    'train_size': float,
    'valid_size': float,
    'stratify': (list, tuple, pd.Series, np.ndarray, None),
    'with_cols': bool,
    'reset_index': bool
})
def train_test_split_bigdata(
        df,
        x_cols,
        y_col,
        shuffle=True,
        return_valid=True,
        random_state=42,
        train_size=0.8,
        valid_size=0.5,
        stratify=None,
        with_cols=False,
        reset_index=True
):
    """切割大型数据集
    
    默认返回六个数据集: X_train, X_valid, X_test, y_train, y_valid, y_test

    训练集、验证集、测试集默认比例为 8:1:1

    :params:
    with_cols: 是否返回切割后的pandas.DataFrame, default to False, 返回numpy.ndarray
    """
    res = train_test_split_bigdata_df(
        df=df,
        x_cols=x_cols,
        y_col=y_col,
        shuffle=shuffle,
        return_valid=return_valid,
        random_state=random_state,
        train_size=train_size,
        valid_size=valid_size,
        stratify=stratify,
        reset_index=reset_index
    )

    if not return_valid:
        train_df, test_df = res
        if with_cols:
            return train_df[x_cols], test_df[x_cols], \
                train_df[y_col].values, test_df[y_col].values

        return train_df[x_cols].values, test_df[x_cols].values, \
            train_df[y_col].values, test_df[y_col].values

    train_df, valid_df, test_df = res

    if with_cols:
        return train_df[x_cols], \
            valid_df[x_cols], \
            test_df[x_cols], \
            train_df[y_col].values, valid_df[y_col].values, test_df[y_col].values

    return train_df[x_cols].values, valid_df[x_cols].values, test_df[x_cols].values, \
        train_df[y_col].values, valid_df[y_col].values, test_df[y_col].values


@TypeAssert({
    'df': (pd.DataFrame, np.ndarray),
    'rows_limit': int
})
def df_block_split(df, rows_limit=10000):
    """
    将pandas.core.dataframe切割，并返回索引生成器
    """
    import numpy as np

    if df.shape[0] <= rows_limit:
        yield df.index.values
    else:
        split_num = int(np.ceil(df.shape[0] / rows_limit))

        idx = df.index.tolist()
        indices = [(index, value) for index, value in enumerate(idx)]

        for i in range(split_num):
            yield [i[1] for i in indices[i * rows_limit: (i + 1) * rows_limit]]
