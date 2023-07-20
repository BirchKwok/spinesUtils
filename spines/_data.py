"""数据类"""
from tqdm.auto import tqdm


def dataloader(
    file_root_path, name_prefix, sep=',',
    chunk_size=None, save_as_pkl=True, transform2low_mem=True,
    search_pkl_first=True, use_dask=False, pandas_read_csv_kwargs=None
):
    """
    :params:
    file_root_path: 文件所在父文件夹
    name_prefix: 文件名，如文件为test.csv, 则传入test，函数会自动寻找文件
    sep: 文件分隔符
    chunksize: 每次读取行数，如果为None，默认一次性全部读取，否则将分批加载进内存
    """
    assert chunk_size is None or isinstance(chunk_size, int)
    assert pandas_read_csv_kwargs is None or isinstance(pandas_read_csv_kwargs, dict)
    import os
    import pandas as pd

    pkl_name = os.path.join(file_root_path, name_prefix+'.pkl')
    tsv_name = os.path.join(file_root_path, name_prefix+'.tsv')
    csv_name = os.path.join(file_root_path, name_prefix + '.csv')
    txt_name = os.path.join(file_root_path, name_prefix + '.txt')

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

        print("File found: ", name)

        if chunk_size:
            data = pd.DataFrame()
            for _tdf in tqdm(pd.read_csv(name, sep=sep, chunksize=chunk_size), desc="Loading..."):
                data = pd.concat((data, _tdf), axis=0)
        else:
            if use_dask:
                import dask.dataframe as dd
                ddf = dd.read_csv(name, sep=sep, assume_missing=True)
                data = ddf.compute().reset_index(drop=True)
            else:
                if pandas_read_csv_kwargs is not None:
                    if 'filepath_or_buffer' in pandas_read_csv_kwargs.keys():
                        del pandas_read_csv_kwargs['filepath_or_buffer']
                    if 'sep' in pandas_read_csv_kwargs.keys():
                        del pandas_read_csv_kwargs['sep']

                if pandas_read_csv_kwargs is not None and len(pandas_read_csv_kwargs) > 0:
                    data = pd.read_csv(name, sep=sep, **pandas_read_csv_kwargs)
                else:
                    data = pd.read_csv(name, sep=sep)
        
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
        from ._preprocessing import transform_dtypes_low_mem
        transform_dtypes_low_mem(df)

    if save_as_pkl:
        df.to_pickle(pkl_name)

    return df
