import re
from functools import wraps, reduce

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

from spinesUtils.asserts import generate_function_kwargs, TypeAssert
from spinesUtils.utils import Printer


def show_mem_change(func):
    def reduce_dataset_mem(dataset):
        if isinstance(dataset, (list, tuple)):
            return reduce(lambda x, y: x + y,
                          map(lambda s: s.memory_usage(deep=True).sum() / 1024 ** 2, dataset))
        return dataset.memory_usage(deep=True).sum() / 1024 ** 2

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = Printer(with_time=False)
        kwargs = generate_function_kwargs(func, args, kwargs)
        if isinstance(kwargs.get('dataset'), (list, tuple)):
            assert all([isinstance(i, pd.DataFrame) for i in kwargs['dataset']])

        start_mem = reduce_dataset_mem(kwargs.get('dataset')) or 0
        res = func(**kwargs)
        end_mem = reduce_dataset_mem(res if res is not None else kwargs.get('dataset')) or 0

        if kwargs.get('verbose'):
            logger.print(f'Memory usage before conversion is: {round(start_mem, 2)} MB  ')
            logger.print(f'Memory usage after conversion is: {round(end_mem, 2)} MB  ')
            try:
                decrease = round(100 * (start_mem - end_mem) / start_mem, 1)
            except ZeroDivisionError:
                decrease = 0

            logger.print(
                f'After conversion, the percentage of memory fluctuation is'
                f' {decrease} %'
            )
        return res

    return wrapper


@show_mem_change
def transform_dtypes_low_mem2(dataset, verbose=True, inplace=True):
    """压缩数据, 使用pandas默认方式"""

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
def transform_dtypes_low_mem(dataset, verbose=True, inplace=True):
    """压缩数据"""

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
def transform_batch_dtypes_low_mem(dataset, verbose=True, inplace=True):
    """批量缩小pandas.core.dataframe的内存占用"""
    assert isinstance(dataset, (list, tuple))

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
def inverse_transform_dtypes(dataset, verbose=True, int_dtypes=np.int32, float_dtypes=np.float32, inplace=True):
    """
    还原为python格式
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
            if float_dtypes is None:
                float_dtypes = float
            dtypes[k] = float_dtypes
        elif str(dtypes[k]).startswith('int') and not isinstance(dtypes[k], int):
            if int_dtypes is None:
                int_dtypes = int
            dtypes[k] = int_dtypes

    ds.__dict__.update(ds.astype(dtypes).__dict__)

    return ds if not inplace else None


@show_mem_change
def inverse_transform_batch_dtypes(dataset, verbose=True, int_dtypes=np.int32, float_dtypes=np.float32, inplace=True):
    """
    批量还原为python格式
    """
    assert isinstance(dataset, (list, tuple))

    if verbose:
        from functools import partial
        ranger = partial(trange, desc="Batch converting ...")
    else:
        ranger = range

    res = []
    for idx in ranger(len(dataset)):
        res.append(inverse_transform_dtypes(dataset[idx], int_dtypes=int_dtypes,
                                            float_dtypes=float_dtypes, verbose=False, inplace=inplace))

    return res if not inplace else None
