"""数据类"""
from tqdm.auto import tqdm

from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils._memory_reduction import memory_reduction


@ParameterTypeAssert({
    'fp': str,
    'sep': str,
    'chunk_size': (None, int),
    'transform2low_mem': bool,
    'turbo_method': str,
    'encoding': (None, str),
    'verbose': bool
})
@ParameterValuesAssert({
    'turbo_method': ('pandas', 'pyarrow', 'dask', 'polars')
})
@memory_reduction
def read_csv(
        fp,
        sep=',',
        chunk_size=None,
        transform2low_mem=True,
        turbo_method='polars',
        encoding='utf8',
        verbose=False,
        **read_csv_kwargs
):
    """
    :params:
    fp: 文件路径
    sep: 文件分隔符
    chunk_size: 每次读取行数，如果为None，默认一次性全部读取，否则将分批加载进内存
    """
    def turbo_reader(fpath, tm):
        if tm == 'polars':
            from polars import read_csv
            return read_csv(fpath, separator=sep, encoding=encoding, **read_csv_kwargs).to_pandas()

        if tm == 'pyarrow':
            from pyarrow.csv import read_csv, ParseOptions, ReadOptions
            return read_csv(fpath,
                            read_options=ReadOptions(encoding=encoding),
                            parse_options=ParseOptions(delimiter=sep), **read_csv_kwargs).to_pandas()

        if tm == 'dask':
            from dask.dataframe import read_csv
            return read_csv(
                fpath, sep=sep, assume_missing=True,
                encoding=encoding, **read_csv_kwargs).compute().reset_index(drop=True)

        if tm == 'pandas':
            from pandas import read_csv
            return read_csv(fpath, sep=sep, encoding=encoding, **read_csv_kwargs)

    def read_text(name):
        import pandas as pd
        data = pd.DataFrame()
        if not verbose:
            iter_pieces = tqdm(pd.read_csv(name, sep=sep, chunksize=chunk_size, encoding=encoding,
                                           **read_csv_kwargs), desc="Loading")
        else:
            iter_pieces = pd.read_csv(name, sep=sep, chunksize=chunk_size, encoding=encoding, **read_csv_kwargs)

        for _tdf in iter_pieces:
            if transform2low_mem:
                from ..preprocessing import transform_dtypes_low_mem
                transform_dtypes_low_mem(_tdf, verbose=False, inplace=True)
            data = pd.concat((data, _tdf), axis=0)

        return data

    if chunk_size:
        df = read_text(fp)
    else:
        df = turbo_reader(fp, tm=turbo_method)

    if transform2low_mem:
        from ..preprocessing import transform_dtypes_low_mem
        transform_dtypes_low_mem(df, verbose=verbose, inplace=True)

    return df
