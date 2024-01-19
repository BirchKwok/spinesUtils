from tqdm.auto import tqdm

from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert


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
def read_csv(fp, sep=',', chunk_size=None, transform2low_mem=False, turbo_method='polars',
             encoding='utf8', verbose=False, **read_csv_kwargs):
    """
    Read a CSV file into a pandas DataFrame with optional memory optimization and using various high-speed methods.

    This function provides an interface to read CSV files using multiple libraries including pandas,
    Polars, PyArrow, and Dask, with options to read in chunks and transform data types to optimize memory usage.

    Parameters
    ----------
    fp : str
        File path or object to read.
    sep : str, default=','
        String of length 1. Field delimiter for the output file.
    chunk_size : int or None, default=None
        Number of rows to read at a time. This is to enable reading large files in chunks.
    transform2low_mem : bool, default=False
        If set to True, transform data types into memory efficient ones using custom logic.
    turbo_method : str, default='polars'
        The library to use for reading the CSV file. Options are 'pandas', 'pyarrow', 'dask', 'polars'.
    encoding : str or None, default='utf8'
        Encoding to use for UTF when reading/writing (ex. 'utf-8').
    verbose : bool, default=False
        If set to True, will print progress messages.
    **read_csv_kwargs : additional keyword arguments
        Additional keyword arguments to pass to the CSV reader function.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the data read from the CSV file.

    Examples
    --------
    >>> df = read_csv('data.csv', sep=',', chunk_size=10000,
    ...    transform2low_mem=True, turbo_method='pandas', verbose=True)
    >>> df.head()
    # Output would display the first 5 rows of the DataFrame read from 'data.csv'.

    Notes
    -----
    - The 'pandas' method is the standard CSV reader.
    - The 'polars', 'pyarrow', and 'dask' methods are high-speed alternatives that may offer performance benefits.
    - Reading in chunks (`chunk_size`) is useful for large files that do not fit into memory.
    - Memory optimization (`transform2low_mem`) involves converting data types of columns to
        more memory-efficient types.
    """
    def turbo_reader(fpath, tm):
        if tm == 'polars':
            from polars import read_csv
            return read_csv(fpath, separator=sep, encoding=encoding, **read_csv_kwargs).to_pandas()

        if tm == 'pyarrow':
            from pyarrow.csv import read_csv, ParseOptions, ReadOptions
            return read_csv(fpath, read_options=ReadOptions(encoding=encoding),
                            parse_options=ParseOptions(delimiter=sep), **read_csv_kwargs).to_pandas()

        if tm == 'dask':
            from dask.dataframe import read_csv
            return read_csv(fpath, sep=sep, assume_missing=True, encoding=encoding,
                            **read_csv_kwargs).compute().reset_index(drop=True)

        if tm == 'pandas':
            from pandas import read_csv
            return read_csv(fpath, sep=sep, encoding=encoding, **read_csv_kwargs)

    def read_text(name):
        import pandas as pd
        if verbose:
            iter_pieces = pd.read_csv(name, sep=sep, chunksize=chunk_size, encoding=encoding, **read_csv_kwargs)
        else:
            iter_pieces = tqdm(pd.read_csv(name, sep=sep, chunksize=chunk_size,
                                           encoding=encoding, **read_csv_kwargs), desc="Loading")

        data_chunks = []
        for _tdf in iter_pieces:
            if transform2low_mem:
                from ..preprocessing import transform_dtypes_low_mem
                transform_dtypes_low_mem(_tdf, verbose=False, inplace=True)
            data_chunks.append(_tdf)

        return pd.concat(data_chunks, axis=0)

    if chunk_size:
        df = read_text(fp)
    else:
        df = turbo_reader(fp, turbo_method)

    if transform2low_mem:
        from ..preprocessing import transform_dtypes_low_mem
        transform_dtypes_low_mem(df, verbose=verbose, inplace=True)

    return df
