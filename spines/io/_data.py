"""数据类"""
from tqdm.auto import tqdm


def dataloader(
        fp=None, file_root_path=None, name_prefix=None, sep=',',
        chunk_size=None, save_as_pkl=True, transform2low_mem=True,
        search_pkl_first=True, turbo_method='auto',
        encoding='utf8', pandas_read_csv_kwargs=None
):
    """
    :params:
    fp: 文件路径
    file_root_path: 文件所在父文件夹
    name_prefix: 文件名，如文件为test.csv, 则传入test，函数会自动寻找文件
    sep: 文件分隔符
    chunk_size: 每次读取行数，如果为None，默认一次性全部读取，否则将分批加载进内存
    """
    assert chunk_size is None or isinstance(chunk_size, int)
    assert pandas_read_csv_kwargs is None or isinstance(pandas_read_csv_kwargs, dict)
    assert encoding is None or isinstance(encoding, str)
    import os
    import pandas as pd

    if fp:
        (file_root_path, name_prefix) = os.path.split(fp)
        name_prefix = ''.join(name_prefix.split('.')[:-1])

    pkl_name = os.path.join(file_root_path, name_prefix + '.pkl')
    tsv_name = os.path.join(file_root_path, name_prefix + '.tsv')
    csv_name = os.path.join(file_root_path, name_prefix + '.csv')
    txt_name = os.path.join(file_root_path, name_prefix + '.txt')

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
            print("File found: ", pkl_name)

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
            for _tdf in tqdm(pd.read_csv(name, sep=sep, chunksize=chunk_size, encoding=encoding), desc="Loading..."):
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
        transform_dtypes_low_mem(df)

    if save_as_pkl:
        df.to_pickle(pkl_name)

    return df
