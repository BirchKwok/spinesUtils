import numpy as np
import pandas as pd
from pandas import DataFrame

from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert


@ParameterTypeAssert({'df': DataFrame, 'threshold': float})
@ParameterValuesAssert({'threshold': 'lambda s: 0 <= s <= 1'})
def variation_threshold(df, threshold=0.01):
    """
    Select features based on the variation ratio, which is the ratio of the most common value to all observations.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the features to be evaluated.
    threshold : float
        The threshold for the variation ratio to decide feature selection.

    Returns
    -------
    list
        A list of column names that meet the variation ratio threshold.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2], 'B': [1, 2, 3, 4]})
    >>> variation_threshold(df, threshold=0.75)
    ['A']
    """
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
    """
    Select features based on their variance. Features with variance less than or equal to the threshold are selected.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the features to be evaluated.
    threshold : float
        The threshold for the variance below which features are selected.

    Returns
    -------
    list
        A list of column names that meet the variance threshold.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 1], 'B': [1, 2, 3, 4]})
    >>> vars_threshold(df, threshold=0.0)
    ['A']
    """
    df_var = {}
    for i in select_numeric_cols(df):
        df_var[i] = df[i].var()

    cols = []
    for col, v in df_var.items():
        if v <= threshold:
            cols.append(col)

    return cols


@ParameterTypeAssert({
    'df': pd.DataFrame,
    'exclude_binary_value_column': bool
})
def select_numeric_cols(df, exclude_binary_value_column=False):
    """
    Select columns with numeric data types from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame from which to select numeric columns.
    exclude_binary_value_column : bool, default=False
        If True, excludes columns with binary values.

    Returns
    -------
    Index
        An Index of column names that have numeric data types.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [True, False, True]})
    >>> select_numeric_cols(df)
    Index(['A', 'C'], dtype='object')
    """
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
    Quickly select column names with specified data types from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame from which to select columns.
    types : str or list
        The data type(s) to filter for column selection.

    Returns
    -------
    list
        A list of column names that match the specified data types.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [True, False, True]})
    >>> get_specified_type_cols(df, types='int64')
    ['A']
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
    """
    Get the column names for features (X) from a DataFrame, optionally excluding specified columns.

    Parameters
    ----------
    df : DataFrame
        The DataFrame from which to select feature columns.
    y_col : str or list
        The column name(s) representing the target variable(s).
    exclude_cols : str or list, optional
        The column name(s) to exclude from the feature columns.

    Returns
    -------
    list
        A list of column names to be used as features.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'target': [7, 8, 9]})
    >>> get_x_cols(df, 'target')
    ['A', 'B']
    """
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
    """
    Return a DataFrame excluding specified columns.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to process.
    cols : list, tuple, ndarray, Series
        The column(s) to exclude.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns excluded.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> exclude_columns(df, ['B', 'C'])
       A
    0  1
    1  2
    2  3
    """
    column_num = df.shape[1]

    if len(cols) / column_num < 0.3:
        return df.drop(columns=cols)

    return df.loc[:, ~df.columns.isin(cols)]
