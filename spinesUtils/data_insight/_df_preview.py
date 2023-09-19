import pandas as pd
import numpy as np

from spinesUtils.feature_tools import select_numeric_cols
from spinesUtils.asserts import TypeAssert


@TypeAssert({'df': pd.DataFrame, 'target_col': str, 'groupby': (None, str)})
def classify_samples_dist(df, target_col, groupby=None):
    """查看样本分布情况"""
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


@TypeAssert({'df': pd.DataFrame})
def show_na_inf(df):
    """各列空值、无穷值分布"""
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


@TypeAssert({'dataset': pd.DataFrame, 'indicators': (None, list, tuple)})
def df_preview(dataset, indicators=None):
    """
    For data previews, to check various data properties of the dataset.
    returns:
    total: number of elements
    na: null values
    naPercent: null value accounts for this ratio
    nunique: unique values number
    dtype: datatype
    75%: 75% quantile
    25%: 25% quantile
    variation: variation ratio
    std: standard deviation
    skew: Skewness
    kurt: Kurtosis
    samples: Random returns two values
    """

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

                global_params['na'] = na_sum
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

        if 'skew' in indicators:
            global_params['skew'] = i_data.skew() if i in num_cols else np.nan

        if 'kurt' in indicators:
            global_params['kurt'] = i_data.kurt() if i in num_cols else np.nan

        df.loc[i] = pd.Series({k: global_params[k] for k in indicators}, name=i)

    if df.shape[0] < df.shape[1]:
        return df.T
    else:
        return df


@TypeAssert({'df': pd.DataFrame})
def df_simple_view(df):
    """仅浏览数值大小分布"""
    numeric_cols = select_numeric_cols(df)
    return df[numeric_cols].describe().T.sort_values('std', ascending=False) \
        .style.bar(subset=['mean'], color='#7BCC70') \
        .background_gradient(subset=['std'], cmap='Reds') \
        .background_gradient(subset=['50%'], cmap='coolwarm')
