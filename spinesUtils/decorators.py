import time
import warnings
from functools import wraps, partial

from collections import OrderedDict


def retry(max_tries=3, delay_seconds=1):
    """
    Decorator that allows a function to retry execution a specified number of times with a delay between each try.

    Parameters
    ----------
    max_tries : int, optional (default=3)
        The maximum number of attempts to execute the function.
    delay_seconds : int, optional (default=1)
        The number of seconds to wait between each retry.

    Returns
    -------
    function
        The decorated function with retry logic.

    Examples
    --------
    >>> @retry(max_tries=4, delay_seconds=2)
    ... def might_fail():
    ...     # Function logic that might raise an exception
    ...     pass
    """
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)

        return wrapper_retry

    return decorator_retry


def memoize(max_size=128):
    """
    Decorator that caches the results of a function call. Once the cache reaches its maximum size,
    the oldest results are discarded.

    Parameters
    ----------
    max_size : int, optional (default=128)
        The maximum number of results to store in the cache.

    Returns
    -------
    function
        The decorated function with memoization.

    Examples
    --------
    >>> @memoize(max_size=100)
    ... def expensive_computation(*args):
    ...     # Function logic that is computationally expensive
    ...     pass
    """
    cache = OrderedDict()
    miss = object()

    def clear_cache():
        cache.clear()

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            result = cache.get(key, miss)
            if result is miss:
                result = fn(*args, **kwargs)
                cache[key] = result

                if len(cache) > max_size:
                    cache.popitem(last=False)

            return result

        wrapper.clear_cache = clear_cache
        return wrapper

    decorator.clear_cache = clear_cache
    return decorator


def timing_decorator(func):
    """
    Decorator that measures and prints the execution time of a function.

    Parameters
    ----------
    func : function
        The function to be measured.

    Returns
    -------
    function
        The decorated function with timing.

    Examples
    --------
    >>> @timing_decorator
    ... def some_function():
    ...     # Function logic
    ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


def log_execution(func):
    """
    Decorator that logs the start and end of a function's execution.

    Parameters
    ----------
    func : function
        The function to be logged.

    Returns
    -------
    function
        The decorated function with logging.

    Examples
    --------
    >>> @log_execution
    ... def some_function():
    ...     # Function logic
    ...     pass
    """
    from spinesUtils.logging import Logger
    logger = Logger(name=func.__name__,
                    level="INFO", with_time=True, use_utc_time=False)

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished executing {func.__name__}")
        return result

    return wrapper


def deprecated(func=None, message=None):
    """
    Decorator that marks a function as deprecated, issuing a warning when it is called.

    Parameters
    ----------
    func : function, optional
        The function to be marked as deprecated.
    message : str, optional
        A custom message for the deprecation warning.

    Returns
    -------
    function
        The decorated function with deprecation warning.

    Examples
    --------
    >>> @deprecated(message="use new_function() instead")
    ... def old_function():
    ...     # Function logic
    ...     pass
    """
    if func is None:
        return partial(deprecated, message=message)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if message is None:
            warnings.warn(f"Function {func.__name__} is deprecated.", category=DeprecationWarning)
        else:
            warnings.warn(message, category=DeprecationWarning)
        return func(*args, **kwargs)

    return wrapper
