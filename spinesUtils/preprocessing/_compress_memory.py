import re
import gc
import time
from functools import wraps, reduce

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

from spinesUtils.asserts import generate_function_kwargs, ParameterTypeAssert
from spinesUtils.logging import Logger


def show_mem_change(func):
    """
    Decorator that logs the change in memory usage before and after the execution of a function.
    It is designed to work with functions that manipulate pandas DataFrames, especially
    those that aim to reduce memory usage. The decorator assumes that the 'dataset' argument
    of the wrapped function is either a pandas DataFrame or a list/tuple of DataFrames.

    Parameters
    ----------
    func : function
        The function to be decorated. This function should have a 'dataset' parameter
        and optionally a 'verbose' parameter.

    Returns
    -------
    function
        A wrapper function that adds memory logging capabilities to the original function.

    Notes
    -----
    The decorator uses a custom `Logger` class for logging and `generate_function_kwargs`
    from `spinesUtils.asserts` to process arguments. Memory usage is calculated in MB.

    Examples
    --------
    >>> @show_mem_change
    ... def sample_function(dataset, verbose=True):
    ...     # Function logic here
    ...     return dataset
    ...
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
    >>> sample_function(df)
    Memory usage before conversion is: 0.08 MB
    Memory usage after conversion is: 0.08 MB
    After conversion, the percentage of memory fluctuation is 0.0 %
    """
    def reduce_dataset_mem(dataset):
        if isinstance(dataset, (list, tuple)):
            return reduce(lambda x, y: x + y,
                          map(lambda s: s.memory_usage(deep=True).sum() / 1024 ** 2, dataset))
        return dataset.memory_usage(deep=True).sum() / 1024 ** 2

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = Logger(with_time=False)
        kwargs = generate_function_kwargs(func, *args, **kwargs)
        if isinstance(kwargs.get('dataset'), (list, tuple)):
            assert all([isinstance(i, pd.DataFrame) for i in kwargs['dataset']])

        start_mem = reduce_dataset_mem(kwargs.get('dataset')) or 0
        res = func(**kwargs)
        end_mem = reduce_dataset_mem(res if res is not None else kwargs.get('dataset')) or 0

        if kwargs.get('verbose'):
            logger.info(f'Memory usage before conversion is: {round(start_mem, 2)} MB  ')
            logger.info(f'Memory usage after conversion is: {round(end_mem, 2)} MB  ')
            try:
                decrease = round(100 * (start_mem - end_mem) / start_mem, 1)
            except ZeroDivisionError:
                decrease = 0

            logger.info(
                f'After conversion, the percentage of memory fluctuation is'
                f' {decrease} %'
            )
        return res

    return wrapper


@show_mem_change
@ParameterTypeAssert({
    'dataset': pd.DataFrame,
    'verbose': (bool, int),
    'inplace': bool
})
def transform_dtypes_low_mem2(dataset, verbose=True, inplace=True):
    """
    Compresses data in a pandas DataFrame by converting data types using pandas' default methods,
    aiming to reduce memory usage. The function decorates with @show_mem_change to track memory change.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The DataFrame whose data types are to be transformed.
    verbose : bool, optional (default=True)
        If True, enables the progress output.
    inplace : bool, optional (default=True)
        If True, modifies the DataFrame in place; otherwise, returns a new DataFrame.

    Returns
    -------
    pandas.DataFrame or None
        The transformed DataFrame if inplace is False, otherwise None.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
    >>> new_df = transform_dtypes_low_mem2(df, verbose=False, inplace=False)
    >>> new_df.dtypes
    A      int8
    B    float16
    dtype: object
    """

    if not inplace:
        ds = dataset.copy()
    else:
        ds = dataset

    numerics = ['int16', 'int32', 'int64', 'float32', 'float64', 'int', 'float']

    cols_dtypes = {}
    for k, v in ds.dtypes.to_dict().items():
        if str(v) in numerics:
            cols_dtypes[k] = str(v)

    if verbose:
        iters = tqdm(cols_dtypes.keys(), desc="Converting ...")
    else:
        iters = cols_dtypes.keys()

    def _builtin_func(cols_dtype):
        if re.match('int', cols_dtype):
            return 'integer'
        else:
            return 'float'

    for col in iters:
        _type = _builtin_func(
            cols_dtype=str(cols_dtypes[col])
        )

        ds[col] = pd.to_numeric(ds[col], downcast=_type)

    return ds if not inplace else None


@show_mem_change
@ParameterTypeAssert({
    'dataset': pd.DataFrame,
    'verbose': (bool, int),
    'inplace': bool
})
def transform_dtypes_low_mem(dataset, verbose=True, inplace=True):
    """
    Optimizes memory usage of a pandas DataFrame by converting data types of columns
    to more memory-efficient types based on their content. The function dynamically
    determines the smallest suitable data type for integer and floating-point columns.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The DataFrame whose data types are to be optimized.
    verbose : bool, optional (default=True)
        If True, displays a progress bar during the conversion process.
    inplace : bool, optional (default=True)
        If True, modifies the DataFrame in place; otherwise, returns a new DataFrame.

    Returns
    -------
    pandas.DataFrame or None
        The DataFrame with optimized data types if inplace is False, otherwise None.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
    >>> new_df = transform_dtypes_low_mem(df, verbose=False, inplace=False)
    >>> new_df.dtypes
    A       int8
    B    float16
    dtype: object

    Notes
    -----
    This function uses a custom logic to determine the appropriate data type for each
    numeric column based on the range of values in the column. It considers both integer
    and floating-point types, ranging from np.int8/np.float16 to np.int64/np.float64.
    """

    if not inplace:
        ds = dataset.copy()
    else:
        ds = dataset

    def _builtin_func(arr, cols_dtype, ner):
        c_min = np.min(arr)
        c_max = np.max(arr)

        if re.match('int', cols_dtype):
            if c_min >= ner['np.int8'][0] and c_max <= ner['np.int8'][1]:
                return np.int8
            elif c_min >= ner['np.int16'][0] and c_max <= ner['np.int16'][1]:
                return np.int16
            elif c_min >= ner['np.int32'][0] and c_max <= ner['np.int32'][1]:
                return np.int32
            elif c_min >= ner['np.int64'][0] and c_max <= ner['np.int64'][1]:
                return np.int64
        else:
            if c_min >= ner['np.float16'][0] and c_max <= ner['np.float16'][1]:
                return np.float16
            elif c_min >= ner['np.float32'][0] and c_max <= ner['np.float32'][1]:
                return np.float32
            elif c_min >= ner['np.float64'][0] and c_max <= ner['np.float64'][1]:
                return np.float64

    numerics = ['int16', 'int32', 'int64', 'float32', 'float64', 'int', 'float']
    numerics_extra_range = {
        'np.int8': (np.iinfo(np.int8).min, np.iinfo(np.int8).max),
        'np.int16': (np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        'np.int32': (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        'np.int64': (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        'np.float16': (np.finfo(np.float16).min, np.finfo(np.float16).max),
        'np.float32': (np.finfo(np.float32).min, np.finfo(np.float32).max),
        'np.float64': (np.finfo(np.float64).min, np.finfo(np.float64).max)
    }

    dtypes = ds.dtypes.to_dict()

    target_cols = list(filter(lambda s: s[1] in numerics, ds.dtypes.items()))

    if verbose:
        iters = tqdm(target_cols, desc="Converting ...", total=len(target_cols))
    else:
        iters = target_cols

    for (k, v) in iters:
        _s = ds[k].values

        dtypes[k] = _builtin_func(
            _s, cols_dtype=str(v),
            ner=numerics_extra_range
        )

    ds.__dict__.update(ds.astype(dtypes).__dict__)

    return ds if not inplace else None


@show_mem_change
@ParameterTypeAssert({
    'dataset': (list, tuple),
    'verbose': (bool, int),
    'inplace': bool
})
def transform_batch_dtypes_low_mem(dataset, verbose=True, inplace=True):
    """
    Applies the `transform_dtypes_low_mem` function to each DataFrame in a list or tuple,
    thereby optimizing memory usage of each DataFrame in the batch. This function is
    particularly useful for processing multiple DataFrames in a memory-efficient manner.

    Parameters
    ----------
    dataset : list or tuple of pandas.DataFrame
        A list or tuple containing the DataFrames to be memory-optimized.
    verbose : bool or int, optional (default=True)
        If True or a non-zero integer, shows a progress bar for the batch processing.
    inplace : bool, optional (default=True)
        If True, modifies each DataFrame in the list/tuple in place; otherwise, returns a
        new list/tuple of optimized DataFrames.

    Returns
    -------
    list or tuple or None
        A new list or tuple containing memory-optimized DataFrames if inplace is False,
        otherwise None if the operation is performed in place.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
    >>> df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10.0, 11.0, 12.0]})
    >>> new_data = transform_batch_dtypes_low_mem([df1, df2], verbose=False, inplace=False)
    >>> [df.dtypes for df in new_data]
    [A       int8
     B    float16
     dtype: object,
     C       int8
     D    float16
     dtype: object]

    Notes
    -----
    The function iterates over the list or tuple of DataFrames and applies the
    `transform_dtypes_low_mem` function to each. The `inplace` parameter determines
    whether the original DataFrames are modified or new DataFrames are returned.
    """
    if verbose:
        from functools import partial
        ranger = partial(trange, desc="Batch converting ...")
    else:
        ranger = range

    res = []
    for idx in ranger(len(dataset)):
        res.append(transform_dtypes_low_mem(dataset[idx], verbose=False, inplace=inplace))

    return res if not inplace else None


@show_mem_change
@ParameterTypeAssert({
    'dataset': pd.DataFrame,
    'verbose': (bool, int),
    'int_dtype': (int, np.int8, np.int16, np.int32, np.int64),
    'float_dtype': (float, np.float16, np.float32, np.float64),
    'inplace': bool
})
def inverse_transform_dtypes(dataset, verbose=True, int_dtype=np.int32, float_dtype=np.float32, inplace=True):
    """
    Reverts the data types of a pandas DataFrame to standard Python or numpy data types.
    This function is useful for undoing memory optimization conversions.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The DataFrame whose data types are to be reverted.
    verbose : bool or int, optional (default=True)
        If True or a non-zero integer, shows a progress bar during the conversion process.
    int_dtype : type, optional (default=np.int32)
        The data type to which integer columns are to be converted.
    float_dtype : type, optional (default=np.float32)
        The data type to which float columns are to be converted.
    inplace : bool, optional (default=True)
        If True, modifies the DataFrame in place; otherwise, returns a new DataFrame.

    Returns
    -------
    pandas.DataFrame or None
        The DataFrame with reverted data types if inplace is False, otherwise None.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]}, dtype=np.float16)
    >>> new_df = inverse_transform_dtypes(df, verbose=False, inplace=False)
    >>> new_df.dtypes
    A    int32
    B    float32
    dtype: object
    """

    if not inplace:
        ds = dataset.copy()
    else:
        ds = dataset

    dtypes = ds.dtypes

    if verbose:
        ranger = tqdm(dtypes.keys(), desc="Converting ...")
    else:
        ranger = dtypes.keys()

    dtypes = ds.dtypes.to_dict()

    for k in ranger:
        if str(dtypes[k]).startswith('float') and not isinstance(dtypes[k], float):
            if float_dtype is None:
                float_dtype = float
            dtypes[k] = float_dtype
        elif str(dtypes[k]).startswith('int') and not isinstance(dtypes[k], int):
            if int_dtype is None:
                int_dtype = int
            dtypes[k] = int_dtype

    ds.__dict__.update(ds.astype(dtypes).__dict__)

    return ds if not inplace else None


@show_mem_change
@ParameterTypeAssert({
    'dataset': (list, tuple),
    'verbose': (bool, int),
    'int_dtype': (int, np.int8, np.int16, np.int32, np.int64),
    'float_dtype': (float, np.float16, np.float32, np.float64),
    'inplace': bool
})
def inverse_transform_batch_dtypes(dataset, verbose=True, int_dtype=np.int32, float_dtype=np.float32, inplace=True):
    """
    Applies the `inverse_transform_dtypes` function to each DataFrame in a list or tuple,
    reverting the data types of each DataFrame in the batch to standard types.

    Parameters
    ----------
    dataset : list or tuple of pandas.DataFrame
        A list or tuple containing the DataFrames whose data types are to be reverted.
    verbose : bool or int, optional (default=True)
        If True or a non-zero integer, shows a progress bar for the batch processing.
    int_dtype : type, optional (default=np.int32)
        The data type to which integer columns in all DataFrames are to be converted.
    float_dtype : type, optional (default=np.float32)
        The data type to which float columns in all DataFrames are to be converted.
    inplace : bool, optional (default=True)
        If True, modifies each DataFrame in the list/tuple in place; otherwise, returns
        a new list/tuple of DataFrames with reverted data types.

    Returns
    -------
    list or tuple or None
        A new list or tuple containing DataFrames with reverted data types if inplace is False,
        otherwise None if the operation is performed in place.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3]}, dtype=np.float16)
    >>> df2 = pd.DataFrame({'B': [4.0, 5.0, 6.0]}, dtype=np.float16)
    >>> new_data = inverse_transform_batch_dtypes([df1, df2], verbose=False, inplace=False)
    >>> [df.dtypes for df in new_data]
    [A    int32
     dtype: object,
     B    float32
     dtype: object]
    """
    if verbose:
        from functools import partial
        ranger = partial(trange, desc="Batch converting ...")
    else:
        ranger = range

    res = []
    for idx in ranger(len(dataset)):
        res.append(inverse_transform_dtypes(dataset[idx], int_dtype=int_dtype,
                                            float_dtype=float_dtype, verbose=False, inplace=inplace))

    return res if not inplace else None


def gc_collector(wait_secs=1):
    """
    Decorator that triggers garbage collection after the execution of a function.
    It also provides an option to wait for a specified duration before returning the function's result.

    Parameters
    ----------
    wait_secs : int, optional (default=1)
        The number of seconds to wait after garbage collection and before returning the result.

    Returns
    -------
    function
        A decorator that can be applied to a function to enable automatic garbage collection.

    Examples
    --------
    >>> @gc_collector(wait_secs=2)
    ... def sample_function(x):
    ...     # Function logic here
    ...     return x * 2
    ...
    >>> sample_function(3)
    # Waits for 2 seconds after execution and then returns 6
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            # 清理垃圾
            gc.collect()
            gc.garbage.clear()

            if wait_secs > 0:
                time.sleep(wait_secs)  # 休眠等待

            return res

        return wrapper

    return decorator
