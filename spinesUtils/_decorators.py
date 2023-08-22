import time
from functools import wraps


def retry(max_tries=3, delay_seconds=1):
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


def memoize(fn):
    cache = {}
    miss = object()

    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = cache.get(args, miss)
        if result is miss:
            result = fn(*args, **kwargs)
            cache[args] = result
        return result

    def clear_cache():
        cache.clear()

    wrapper.clear_cache = clear_cache
    return wrapper


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


def log_execution(func):
    import logging

    logging.basicConfig(level=logging.INFO)

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished executing {func.__name__}")
        return result

    return wrapper
