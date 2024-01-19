import numpy as np
import torch

from spinesUtils.asserts import ParameterTypeAssert


@ParameterTypeAssert({
    'data': (np.ndarray, torch.Tensor),
    'condition': bool,
    'shape': tuple
})
def reshape_if(data, condition, shape):
    """
    Reshapes the provided data to a specified shape if a condition is True.
    This function supports both numpy arrays and PyTorch tensors.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        The data to be reshaped.
    condition : bool
        The condition to check before reshaping.
    shape : tuple
        The new shape for the data.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The reshaped data if condition is True, otherwise the original data.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4])
    >>> reshape_if(data, condition=True, shape=(2, 2))
    array([[1, 2],
           [3, 4]])
    """
    if condition:
        return data.reshape(shape)
    return data


@ParameterTypeAssert({
    'data': (np.ndarray, torch.Tensor),
    'condition': bool,
    'dim': (None, int)
})
def squeeze_if(data, condition, dim=None):
    """
    Squeezes the provided data by removing specified dimensions if a condition is True.
    This function supports both numpy arrays and PyTorch tensors.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        The data to be squeezed.
    condition : bool
        The condition to check before squeezing.
    dim : None or int, optional
        The specific dimension to squeeze. If None, all single-dimensional entries are removed.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The squeezed data if condition is True, otherwise the original data.

    Examples
    --------
    >>> data = np.array([[1, 2, 3, 4]])
    >>> squeeze_if(data, condition=True)
    array([1, 2, 3, 4])
    """
    if condition:
        return data.squeeze(dim)
    return data


@ParameterTypeAssert({
    'data': (np.ndarray, torch.Tensor),
    'condition': bool,
    'dim': (None, int)
})
def unsqueeze_if(data, condition, dim=None):
    """
    Un-squeezes the provided data by adding a specified dimension if a condition is True.
    This function supports both numpy arrays and PyTorch tensors.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        The data to be un-squeezed.
    condition : bool
        The condition to check before un-squeezing.
    dim : None or int, optional
        The dimension index at which to un-squeeze. If None, adds a new dimension at index 0.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The un-squeezed data if condition is True, otherwise the original data.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4])
    >>> unsqueeze_if(data, condition=True, dim=1)
    array([[1],
           [2],
           [3],
           [4]])
    """
    if condition:
        if isinstance(data, np.ndarray):
            return np.expand_dims(data, dim)
        return data.unsqueeze(dim)
    return data

