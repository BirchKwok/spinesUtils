import numpy as np
import torch

from spinesUtils.asserts import ParameterTypeAssert, check_obj_is_function, ParameterValuesAssert


@ParameterTypeAssert({
    'data': (np.ndarray, torch.Tensor),
    'shape': tuple
})
@ParameterValuesAssert({
    'condition': lambda s: check_obj_is_function(s)
})
def reshape_if(data, condition, shape):
    """
    Reshape data if condition is true.
    """
    if condition:
        return data.reshape(shape)
    return data


@ParameterTypeAssert({
    'data': (np.ndarray, torch.Tensor),
    'shape': tuple
})
@ParameterValuesAssert({
    'condition': lambda s: check_obj_is_function(s)
})
def squeeze_if(data, condition, dim=None):
    """
    Squeeze data if condition is true.
    """
    if condition:
        return data.squeeze(dim)
    return data


@ParameterTypeAssert({
    'data': (np.ndarray, torch.Tensor),
    'shape': tuple
})
@ParameterValuesAssert({
    'condition': lambda s: check_obj_is_function(s)
})
def unsqueeze_if(data, condition, dim=None):
    """
    Unsqueeze data if condition is true.
    """
    if condition:
        if isinstance(data, np.ndarray):
            return np.expand_dims(data, dim)
        return data.unsqueeze(dim)
    return data

