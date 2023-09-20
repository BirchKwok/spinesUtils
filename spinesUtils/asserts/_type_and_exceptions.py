
class ParametersTypeError(Exception):
    pass

class ParametersValueError(Exception):
    pass


class NotFittedError(Exception):
    pass


def augmented_isinstance(s, types):
    if types is None:
        return s is types
    elif isinstance(types, type):
        return isinstance(s, types)
    elif isinstance(types, tuple):
        without_none_types = tuple(filter(None, types))
        if len(without_none_types) == len(types):
            return isinstance(s, types)
        else:
            return isinstance(s, without_none_types) or (s is None)


def raise_params_numbers_error(func, func_name):
    from ._func_params import get_function_params_name
    raise ParametersTypeError(f"Function {func_name} only "
                              f"accept {len(get_function_params_name(func))} parameter(s)")
