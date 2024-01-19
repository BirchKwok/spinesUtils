import inspect
from inspect import signature


def get_function_params_name(func):
    """
    Retrieve the parameter names of a function and cache the results for future calls.

    Parameters
    ----------
    func : function
        The function for which parameter names are to be retrieved.

    Returns
    -------
    list of str
        A list containing the names of the parameters for the given function.

    Notes
    -----
    This function uses a cache to store the parameter names for functions it has
    already processed to avoid redundant computation in future calls.

    Examples
    --------
    >>> def sample_function(x, y, z=0):
    ...     pass
    >>> get_function_params_name(sample_function)
    ['x', 'y', 'z']
    """
    if not hasattr(get_function_params_name, 'cache'):
        get_function_params_name.cache = {}
    if func not in get_function_params_name.cache:
        get_function_params_name.cache[func] = list(signature(func).parameters.keys())
    return get_function_params_name.cache[func]


def generate_function_kwargs(func, *args, **kwargs):
    """
    Generate a dictionary of keyword arguments for a given function using provided arguments.

    Parameters
    ----------
    func : function
        The function for which the keyword arguments are to be generated.
    *args : tuple
        Positional arguments that would be passed to the function.
    **kwargs : Keyword arguments that would be passed to the function.

    Returns
    -------
    dict
        A dictionary containing the keyword arguments for the function, filled in with
        either provided values or the function's default values.

    Examples
    --------
    >>> def sample_function(a, b, c=10):
    ...     return a + b + c
    >>> generate_function_kwargs(sample_function, 1, 2, c=3)
    {'a': 1, 'b': 2, 'c': 3}
    """
    new_kwargs = {}
    func_params = get_function_params_name(func)
    for args_param, args_value in zip(func_params[:len(args)], args):
        new_kwargs[args_param] = args_value
    for default_param, default_value in signature(func).parameters.items():
        if default_value.default is inspect._empty:
            continue
        elif default_param not in new_kwargs:
            new_kwargs[default_param] = default_value.default
    for k, v in kwargs.items():
        new_kwargs[k] = v
    return new_kwargs


def check_has_param(func, param):
    """
    Check if a function has a given parameter.

    Parameters
    ----------
    func : function
        The function to check for the presence of the parameter.
    param : str
        The name of the parameter to check for.

    Returns
    -------
    bool
        True if the parameter exists in the function's signature, False otherwise.

    Examples
    --------
    >>> def sample_function(x, y):
    ...     pass
    >>> check_has_param(sample_function, 'x')
    True
    >>> check_has_param(sample_function, 'z')
    False
    """
    sig = inspect.signature(func)
    _param = sig.parameters.get(param, None)
    if _param is not None:
        return True
    else:
        return False
