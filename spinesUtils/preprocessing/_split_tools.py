import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from spinesUtils.asserts import ParameterTypeAssert


@ParameterTypeAssert({
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
    """
    Splits a large pandas DataFrame into train, validation, and test sets, with an option to stratify.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be split.
    x_cols : list, tuple, pandas.Series, numpy.ndarray
        The column names to be used as features.
    y_col : str
        The column name to be used as the target variable.
    shuffle : bool, optional (default=True)
        Whether to shuffle the data before splitting.
    return_valid : bool, optional (default=True)
        Whether to return a validation set.
    random_state : None or int, optional (default=42)
        The random state for shuffling.
    train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the train split.
    valid_size : float, optional (default=0.5)
        The proportion of the validation set in the remaining data after the train split.
    stratify : list, tuple, pandas.Series, numpy.ndarray, or None, optional
        The data to use for stratifying the split.
    reset_index : bool, optional (default=True)
        If True, the index of the resulting DataFrames is reset.

    Returns
    -------
    tuple of pandas.DataFrame
        The train, validation, and test DataFrames.

    Examples
    --------
    >>> df = pd.DataFrame({'feature1': range(100), 'feature2': range(100), 'target': range(100)})
    >>> train_df, valid_df, test_df = train_test_split_bigdata_df(df, x_cols=['feature1', 'feature2'], y_col='target', shuffle=True, random_state=42)
    >>> train_df.shape, valid_df.shape, test_df.shape
    ((80, 3), (10, 3), (10, 3))
    """

    idx = np.arange(df.shape[0])
    y = df[y_col].values

    stratify_vals = y if stratify is not None else None

    X_train_idx, X_test_idx, _, _ = train_test_split(
        idx, y,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify_vals
    )

    if not return_valid:
        # 利用索引返回DataFrame
        return df.iloc[X_train_idx][x_cols + [y_col]], df.iloc[X_test_idx][x_cols + [y_col]]

    X_valid_idx, X_test_idx, _, _ = train_test_split(
        X_test_idx, y[X_test_idx],
        train_size=valid_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify_vals if return_valid and stratify is not None else None
    )

    # 利用索引返回DataFrame
    train_df = df.iloc[X_train_idx][x_cols + [y_col]]
    valid_df = df.iloc[X_valid_idx][x_cols + [y_col]]
    test_df = df.iloc[X_test_idx][x_cols + [y_col]]

    if reset_index:
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

    return train_df, valid_df, test_df


@ParameterTypeAssert({
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
    """
    Splits a large pandas DataFrame into arrays or DataFrames for training, validation, and testing.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be split.
    x_cols : list, tuple, pandas.Series, numpy.ndarray
        The column names to be used as features.
    y_col : str
        The column name to be used as the target variable.
    shuffle : bool, optional (default=True)
        Whether to shuffle the data before splitting.
    return_valid : bool, optional (default=True)
        Whether to return a validation set.
    random_state : None or int, optional (default=42)
        The random state for shuffling.
    train_size : float, optional (default=0.8)
        The proportion of the dataset to include in the train split.
    valid_size : float, optional (default=0.5)
        The proportion of the validation set in the remaining data after the train split.
    stratify : list, tuple, pandas.Series, numpy.ndarray, or None, optional
        The data to use for stratifying the split.
    with_cols : bool, optional (default=False)
        If True, return DataFrames with column names; otherwise, return numpy arrays.
    reset_index : bool, optional (default=True)
        If True, the index of the resulting DataFrames is reset.
    Returns
    -------
    tuple of numpy.ndarray or pandas.DataFrame
        The train, validation, and test data as numpy arrays or DataFrames.

    Examples
    --------
    >>> df = pd.DataFrame({'feature1': range(100), 'feature2': range(100), 'target': range(100)})
    >>> train_X, valid_X, test_X, train_y, valid_y, test_y = train_test_split_bigdata(
    ...        df, x_cols=['feature1', 'feature2'], y_col='target', shuffle=True,
    ...         random_state=42, with_cols=True)
    >>> train_X.shape, valid_X.shape, test_X.shape, train_y.shape, valid_y.shape, test_y.shape
    ((80, 2), (10, 2), (10, 2), (80,), (10,), (10,))
    """
    idx = np.arange(df.shape[0])
    y = df[y_col].values

    stratify_vals = y if stratify is not None else None

    # 分割训练集和测试集
    X_train_idx, X_test_idx, _, _ = train_test_split(
        idx, y, train_size=train_size, shuffle=shuffle, random_state=random_state, stratify=stratify_vals)

    if not return_valid:
        # 使用索引获取训练集和测试集
        train_df, test_df = df.iloc[X_train_idx], df.iloc[X_test_idx]
        return _process_split_result(train_df, None, test_df, x_cols, None, y_col, with_cols)

    # 分割验证集和测试集
    X_valid_idx, X_test_idx, _, _ = train_test_split(
        X_test_idx, y[X_test_idx], train_size=valid_size, shuffle=shuffle, random_state=random_state,
        stratify=stratify_vals)

    # 使用索引获取训练集、验证集和测试集
    train_df, valid_df, test_df = df.iloc[X_train_idx], df.iloc[X_valid_idx], df.iloc[X_test_idx]
    return _process_split_result(train_df, valid_df, test_df, x_cols, y_col, with_cols, reset_index)


def _process_split_result(train_df, valid_df, test_df, x_cols, y_col, with_cols, reset_index):
    if with_cols:
        if reset_index:
            # 重置索引仅在必要时进行
            train_df = train_df[x_cols + [y_col]].reset_index(drop=True)
            test_df = test_df[x_cols + [y_col]].reset_index(drop=True)
            valid_df = valid_df[x_cols + [y_col]].reset_index(drop=True) if valid_df is not None else None
        else:
            # 无需重置索引时，直接选择列
            train_df = train_df[x_cols + [y_col]]
            test_df = test_df[x_cols + [y_col]]
            valid_df = valid_df[x_cols + [y_col]] if valid_df is not None else None
    else:
        # 直接使用 NumPy 数组
        train_df = (train_df[x_cols].values, train_df[y_col].values)
        test_df = (test_df[x_cols].values, test_df[y_col].values)
        valid_df = (valid_df[x_cols].values, valid_df[y_col].values) if valid_df is not None else None

    # 减少条件判断，直接构造返回值
    return train_df + (valid_df if valid_df is not None else ()) + test_df


@ParameterTypeAssert({
    'df': (pd.DataFrame, np.ndarray),
    'rows_limit': int
})
def df_block_split(df, rows_limit=10000):
    """
    Splits a DataFrame into blocks of specified row limit.
    Parameters
    ----------
    df : pandas.DataFrame or numpy.ndarray
        The DataFrame or numpy array to be split.
    rows_limit : int, optional (default=10000)
        The maximum number of rows in each block.

    Yields
    ------
    pandas.DataFrame or numpy.ndarray
        Blocks of data with the specified row limit.

    Examples
    --------
    >>> df = pd.DataFrame({'A': range(100)})
    >>> for block in df_block_split(df, rows_limit=20):
    ...     print(block.shape)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    (20, 1)
    >>> # The actual number of yielded blocks depends on the input DataFrame's shape.
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
