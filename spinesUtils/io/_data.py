"""数据类"""
from tqdm.auto import tqdm

from spinesUtils.utils import Printer
from spinesUtils.asserts import TypeAssert


@TypeAssert({
    'fp': (None, str),
    'file_root_path': (None, str),
    'name_prefix': (None, str),
    'sep': str,
    'chunk_size': (None, int),
    'save_as_pkl': bool,
    'transform2low_mem': bool,
    'search_pkl_first': bool,
    'turbo_method': str,
    'encoding': (None, str),
    'verbose': bool,
    'pandas_read_csv_kwargs': (None, dict)
})
def dataloader(
        fp=None, file_root_path=None, name_prefix=None, sep=',',
        chunk_size=None, save_as_pkl=False, transform2low_mem=True,
        search_pkl_first=False, turbo_method='auto',
        encoding='utf8', verbose=False, pandas_read_csv_kwargs=None
):
    """
    :params:
    fp: 文件路径
    file_root_path: 文件所在父文件夹
    name_prefix: 文件名，如文件为test.csv, 则传入test，函数会自动寻找文件
    sep: 文件分隔符
    chunk_size: 每次读取行数，如果为None，默认一次性全部读取，否则将分批加载进内存
    """
    import os
    from pathlib import Path
    import pandas as pd

    if fp:
        (file_root_path, name_prefix) = os.path.split(fp)
        name_prefix = fp[:-len(Path(fp).suffix)]

    pkl_name = os.path.join(file_root_path, name_prefix + '.pkl')
    tsv_name = os.path.join(file_root_path, name_prefix + '.tsv')
    csv_name = os.path.join(file_root_path, name_prefix + '.csv')
    txt_name = os.path.join(file_root_path, name_prefix + '.txt')

    logger = Printer()

    def turbo_reader(fpath, tm):
        if tm == 'auto':
            try:
                try:
                    from pyarrow.csv import read_csv, ParseOptions, ReadOptions
                    tm = 'pyarrow'
                except ImportError:
                    from dask.dataframe import read_csv
                    tm = 'dask'
            except ImportError:
                from pandas import read_csv
                tm = 'pandas'

        if tm == 'pyarrow':
            from pyarrow.csv import read_csv, ParseOptions, ReadOptions
            return read_csv(fpath,
                            read_options=ReadOptions(encoding=encoding),
                            parse_options=ParseOptions(delimiter=sep)).to_pandas()

        if tm == 'dask':
            from dask.dataframe import read_csv
            ddf = read_csv(fpath, sep=sep, assume_missing=True, encoding=encoding)
            return ddf.compute().reset_index(drop=True)

        if tm not in ('auto', 'pyarrow', 'dask'):
            from pandas import read_csv
            return read_csv(fpath, sep=sep, encoding=encoding)

    def read_pkl():
        if os.path.isfile(pkl_name):
            logger.print("File found: ", pkl_name)

            return pd.read_pickle(pkl_name)

        return None

    def read_text():
        if os.path.isfile(tsv_name):
            name = tsv_name
        elif os.path.isfile(csv_name):
            name = csv_name
        elif os.path.isfile(txt_name):
            name = txt_name
        else:
            return None

        if chunk_size:
            data = pd.DataFrame()
            if not verbose:
                iter_pieces = tqdm(pd.read_csv(name, sep=sep, chunksize=chunk_size, encoding=encoding), desc="Loading")
            else:
                iter_pieces = pd.read_csv(name, sep=sep, chunksize=chunk_size, encoding=encoding)

            for _tdf in iter_pieces:
                if transform2low_mem:
                    from ..preprocessing import transform_dtypes_low_mem
                    transform_dtypes_low_mem(_tdf, verbose=False)
                data = pd.concat((data, _tdf), axis=0)
        else:
            if turbo_method:
                data = turbo_reader(name, turbo_method)
            else:
                if pandas_read_csv_kwargs is not None:
                    if 'filepath_or_buffer' in pandas_read_csv_kwargs.keys():
                        del pandas_read_csv_kwargs['filepath_or_buffer']
                    if 'sep' in pandas_read_csv_kwargs.keys():
                        del pandas_read_csv_kwargs['sep']

                if pandas_read_csv_kwargs is not None and len(pandas_read_csv_kwargs) > 0:
                    data = pd.read_csv(name, sep=sep, encoding=encoding, **pandas_read_csv_kwargs)
                else:
                    data = pd.read_csv(name, sep=sep, encoding=encoding)

        return data

    if search_pkl_first:
        df = read_pkl()
        if df is None:
            df = read_text()
    else:
        df = read_text()
        if df is None:
            df = read_pkl()

    if df is None:
        raise FileNotFoundError("The file should have the extension .pkl, .tsv, .csv, or .txt.")

    if transform2low_mem:
        from ..preprocessing import transform_dtypes_low_mem
        transform_dtypes_low_mem(df, verbose=verbose)

    if save_as_pkl:
        df.to_pickle(pkl_name)

    return df
