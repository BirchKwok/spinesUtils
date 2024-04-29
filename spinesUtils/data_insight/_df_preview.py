import pandas as pd
import numpy as np

from spinesUtils.feature_tools import select_numeric_cols
from spinesUtils.asserts import ParameterTypeAssert


@ParameterTypeAssert({'df': pd.DataFrame, 'target_col': str, 'groupby': (None, str)})
def classify_samples_dist(df, target_col, groupby=None):
    """
    Analyzes the distribution of samples in a DataFrame, either overall or grouped by another column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.
    target_col : str
        The name of the column in the DataFrame for which the distribution is calculated.
    groupby : str, optional
        The name of the column to group the data by before calculating the distribution.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the count and rate of each unique value in the target column.

    Examples
    --------
    >>> df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'C'], 'value': [1, 1, 2, 2, 3]})
    >>> classify_samples_dist(df, 'value')
       sample_count   rate
    1             2  40.0%
    2             2  40.0%
    3             1  20.0%
    """
    if groupby is None:
        _ = df[target_col].value_counts()

        res = pd.concat((_, _ / _.sum()), axis=1)
        res.columns = ['sample_count', 'rate']
        res['rate'] = round(res['rate'] * 100, 2).astype(str) + '%'
    else:
        yn = df[target_col].nunique()
        _ = df.groupby(by=[groupby, target_col])[target_col].count()
        _ = pd.DataFrame(_)
        v = df.groupby(by=[groupby])[target_col].count()
        m_index = pd.MultiIndex.from_product([v.index, df[target_col].unique()], names=_.index.names)
        to_div = pd.DataFrame(np.repeat(v.values, yn), index=m_index, columns=[target_col])

        res = pd.concat((_, _ / to_div, _ / _.sum()), axis=1)
        res.columns = ['sample_count', 'group_rate', 'total_rate']
        res['group_rate'] = round(res['group_rate'] * 100, 2).astype(str) + '%'
        res['total_rate'] = round(res['total_rate'] * 100, 2).astype(str) + '%'

    return res


@ParameterTypeAssert({'df': pd.DataFrame})
def show_na_inf(df):
    """
    Displays the distribution of NaN and infinity values across all columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for NaN and infinity values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the counts and percentages of NaN and infinity values per column.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, np.nan, np.inf], 'B': [np.nan, np.nan, 3]})
    >>> show_na_inf(df)
        columns nan_percent nan_count inf_percent inf_count
    A        A        33.33%         1       33.33%        1
    B        B        66.67%         2         0.0%        0
    """
    df_nan_cnt = df.isna().sum(axis=0)
    df_inf_cnt = (df == np.inf).sum(axis=0)

    _ = (df_nan_cnt / df.shape[0] * 100) \
        .to_frame(name='nan_percent') \
        .apply(lambda s: round(s, 4)) \
        .reset_index(drop=False)

    _.columns = ['columns', 'nan_percent']

    _['nan_percent'] = _['nan_percent'].astype(str) + '%'

    _2 = (df_inf_cnt / df.shape[0] * 100) \
        .to_frame(name='inf_percent') \
        .apply(lambda s: round(s, 4)) \
        .reset_index(drop=False)

    _2.columns = ['columns', 'inf_percent']

    _2['inf_percent'] = _2['inf_percent'].astype(str) + '%'

    _ = pd.merge(left=_, right=_2, on='columns', how='left')

    _ = pd.concat(
        (_.set_index('columns'), df_nan_cnt.to_frame(name='nan_count'), df_inf_cnt.to_frame(name='inf_count')
         ), axis=1)

    return _.query("nan_count > 0 | inf_count > 0").sort_values(
        by=['nan_count', 'inf_count'], ascending=False)


@ParameterTypeAssert({'dataset': pd.DataFrame, 'indicators': (None, list, tuple)})
def df_preview(dataset, indicators=None):
    """
    Provides a statistical summary of the dataset with an option to specify which metrics to include.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The DataFrame to summarize.
    indicators : list of str, optional
        The statistical metrics to include in the summary.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the specified statistical metrics for each column.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_preview(df, indicators=['mean', 'std'])
       mean   std
    A   2.0   1.0
    B   5.0   1.0
    """
    from numba import jit

    @jit(nopython=True, fastmath=True)
    def calculate_skewness_kurtosis(data, mean, std):
        var = std ** 2
        # std = np.sqrt(var)

        # 计算三阶和四阶中心矩
        third_moment = np.mean((data - mean) ** 3)
        fourth_moment = np.mean((data - mean) ** 4)

        # 计算偏度和峰度
        skewness = third_moment / (std ** 3)
        kurtosis = fourth_moment / (var ** 2) - 3

        return skewness, kurtosis

    available_indicators = (
        'total', 'na', 'naPercent', 'nunique', 'dtype', 'max', '75%', 'median',
        '25%', 'min', 'mean', 'mode', 'variation', 'std', 'skew', 'kurt', 'samples'
    )

    if indicators is not None:
        unavailable_indicators = []
        for i in indicators:
            if i not in available_indicators:
                unavailable_indicators.append(i)

        if len(unavailable_indicators) > 0:
            raise ValueError(f"{','.join(unavailable_indicators)} is unavailable indicators")

    col = dataset.columns
    num_cols = select_numeric_cols(dataset)

    ind_len = dataset.shape[0]

    dtypes = dataset.dtypes.to_dict()

    if indicators is None:
        indicators = available_indicators

    df = pd.DataFrame(columns=indicators)
    random_idx = np.random.randint(ind_len, size=2)

    global_params = {'total': ind_len}

    for i in col:
        i_data = dataset[i]

        if any([True if ind in indicators else False for ind in
                ['na', 'naPercent', 'nunique', 'max', '75%', 'median',
                 '25%', 'min', 'mean', 'mode', 'variation', 'std']
                ]):
            desc = i_data.describe()

            global_params['nunique'] = i_data.nunique()
            global_params['max'] = desc.get('max')
            global_params['75%'] = desc.get('75%')
            global_params['median'] = desc.get('50%')
            global_params['25%'] = desc.get('25%')
            global_params['min'] = desc.get('min')
            global_params['mean'] = desc.get('mean')
            global_params['std'] = desc.get('std')

            if 'na' in indicators or 'naPercent' in indicators:
                na_sum = ind_len - desc['count']

                global_params['na'] = int(na_sum)
                global_params['naPercent'] = na_sum / ind_len

            if 'mode' in indicators or 'variation' in indicators:
                if not desc.get('top'):
                    _ = i_data.value_counts(ascending=False)
                    mode = _.index[0]
                    mode_count = _.iloc[0]
                else:
                    mode = desc['top']
                    mode_count = desc['freq']

                global_params['mode'] = mode
                global_params['mode_count'] = mode_count
                global_params['variation'] = 1 - mode_count / ind_len

        if 'dtype' in indicators:
            global_params['dtype'] = dtypes[i]

        if 'samples' in indicators:
            global_params['samples'] = tuple([i_data[idx] for idx in random_idx])

        if 'skew' in indicators and 'kurt' in indicators:
            if i not in num_cols:
                global_params['skew'] = None
                global_params['kurt'] = None
            else:
                global_params['skew'], global_params['kurt'] = calculate_skewness_kurtosis(
                    i_data.values, global_params['mean'], global_params['std'])
        elif 'skew' in indicators:
            global_params['skew'] = i_data.skew() if i in num_cols else None
        elif 'kurt' in indicators:
            global_params['kurt'] = i_data.kurt() if i in num_cols else None

        df.loc[i] = pd.Series({k: global_params[k] for k in indicators}, name=i)

    return df


@ParameterTypeAssert({'df': pd.DataFrame})
def df_simple_view(df):
    """
    Provides a simple statistical view of the numeric columns in the DataFrame, highlighting the mean and standard deviation.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to visualize.

    Returns
    -------
    pandas.io.formats.style.Styler
        A Styler object that can be rendered in Jupyter Notebooks to display the DataFrame with conditional formatting.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_simple_view(df)
    # This will return a styled DataFrame when used in a Jupyter Notebook.
    """
    numeric_cols = select_numeric_cols(df)
    return df[numeric_cols].describe().T.sort_values('std', ascending=False) \
        .style.bar(subset=['mean'], color='#7BCC70') \
        .background_gradient(subset=['std'], cmap='Reds') \
        .background_gradient(subset=['50%'], cmap='coolwarm')
