
class ParametersTypeError(Exception):
    """
    Exception raised when the parameters passed to a function do not match the expected types.
    """
    pass


class ParametersValueError(Exception):
    """
    Exception raised when the values of the parameters passed to a function do not match the expected constraints.
    """
    pass


class NotFittedError(Exception):
    """
    Exception raised when trying to use an estimator which has not been fitted yet.
    """
    pass


def augmented_isinstance(s, types):
    """
    Enhanced isinstance check that can handle None as a type option.

    Parameters
    ----------
    s : any
        The object to check.
    types : type or tuple of types or None
        The type(s) to check against. If None, the check is whether 's' is None.

    Returns
    -------
    bool
        True if 's' is an instance of 'types', False otherwise.

    Examples
    --------
    >>> augmented_isinstance(5, int)
    True
    >>> augmented_isinstance(None, (int, None))
    True
    >>> augmented_isinstance('test', str)
    True
    """
    if types is None:
        return s is None
    if isinstance(types, type):
        return isinstance(s, types)
    if isinstance(types, tuple):
        if None in types:
            return s is None or isinstance(s, tuple(t for t in types if t is not None))
        else:
            return isinstance(s, types)
    return False


def raise_params_numbers_error(func, func_name):
    """
    Raise an error if the number of parameters provided does not match the function's signature.

    Parameters
    ----------
    func : callable
        The function to check the parameters against.
    func_name : str
        The name of the function, used for the error message.

    Raises
    ------
    ParametersTypeError
        If the number of parameters does not match.

    Examples
    --------
    >>> def sample_function(x, y):
    ...     pass
    >>> raise_params_numbers_error(sample_function, 'sample_function')
    ParametersTypeError: Function sample_function only accept 2 parameter(s)
    """
    from ._func_params import get_function_params_name
    raise ParametersTypeError(f"Function {func_name} only "
                              f"accept {len(get_function_params_name(func))} parameter(s)")


def raise_if(exception, condition, error_msg):
    """
    Raise an exception with a custom error message if a condition is True.

    Parameters
    ----------
    exception : Exception or BaseException or type
        The exception type to raise.
    condition : bool
        The condition to evaluate.
    error_msg : str
        The error message to use for the exception if raised.

    Raises
    ------
    Exception
        The provided exception type if the condition is True.

    Examples
    --------
    >>> raise_if(ValueError, 1 > 0, "One is greater than zero.")
    ValueError: One is greater than zero.
    """

    assert issubclass(exception, BaseException), "Exception must be a subclass of BaseException."

    if condition:
        raise exception(error_msg)


def raise_if_not(exception, condition, error_msg):
    """
    Raise an exception with a custom error message if a condition is False.

    Parameters
    ----------
    exception : Exception or BaseException or type
        The exception type to raise.
    condition : bool
        The condition to evaluate.
    error_msg : str
        The error message to use for the exception if raised.

    Raises
    ------
    Exception
        The provided exception type if the condition is False.

    Examples
    --------
    >>> raise_if_not(ValueError, 0 > 1, "Zero is not greater than one.")
    ValueError: Zero is not greater than one.
    """
    assert issubclass(exception, BaseException), "Exception must be a subclass of BaseException."

    if not condition:
        raise exception(error_msg)

