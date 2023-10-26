import numpy as np

from spinesUtils.asserts import ParameterTypeAssert, check_obj_is_function, ParameterValuesAssert


@ParameterTypeAssert({
    'data': np.ndarray,
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
