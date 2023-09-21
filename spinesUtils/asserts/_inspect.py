from functools import wraps
from types import FunctionType, LambdaType
from typing import Union

from ._type_and_exceptions import (
    ParametersTypeError,
    augmented_isinstance,
    ParametersValueError,
    raise_params_numbers_error
)
from ._func_params import get_function_params_name, generate_function_kwargs


def get_function_all_kwargs(func, func_name, *args, **kwargs):
    input_size = len(args) + len(kwargs)

    kwargs = generate_function_kwargs(func, *args, **kwargs)

    if input_size > len(kwargs):
        raise_params_numbers_error(func, func_name=func_name or func.__name__)

    return kwargs


def check_obj_is_function(obj):
    if not augmented_isinstance(obj, (FunctionType, LambdaType)):
        return False
    return True


class BaseAssert:
    def __init__(self, params_config, func_name=None):
        self._params_config_value_types, self._params_config_value_type_names = self.params_config_value_type()
        self.check_params_config_type(params_config, func_name)
        self.params_config = params_config
        self.func_name = func_name

    def check_params_config_type(self, params_config, func_name):
        if not augmented_isinstance(func_name, (None, str)):
            raise ParametersTypeError(f"Parameter `func_name` must be str or None.")

        if not augmented_isinstance(params_config, dict):
            raise ParametersTypeError("Parameter `params_config` must be dict.")
        for k, v in params_config.items():
            if not augmented_isinstance(k, str):
                raise ParametersTypeError("Keys of params_config must be string.")

            if not augmented_isinstance(v, self._params_config_value_types):
                raise ParametersTypeError(f"Values of params_config must be "
                                          f"{self._params_config_value_type_names}.")

    @staticmethod
    def params_config_value_type(types=tuple, names='tuple'):
        return types, names

    def check_func_params_exists(self, func):
        for p_name in self.params_config:
            if p_name not in get_function_params_name(func):
                raise ParametersTypeError(
                    f"Function '{self.func_name}' not exists `{p_name}` parameter.")

    def __call__(self, func):
        raise NotImplementedError


class ParameterTypeAssert(BaseAssert):
    """
    function parameters type checker

    """

    def __init__(self, params_config, func_name=None):
        super().__init__(params_config, func_name)

    @staticmethod
    def params_config_value_type(types=(tuple, type, None), names='(tuple, class, None)'):
        return types, names

    @staticmethod
    def type2string(tp: Union[type, tuple]):
        if augmented_isinstance(tp, type):
            return tp.__name__ if tp is not None else 'None'
        else:
            return [i.__name__ if i is not None else 'None' for i in tp]

    def __call__(self, func):
        self.func_name = self.func_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.check_func_params_exists(func)
            kwargs = get_function_all_kwargs(func, self.func_name, *args, **kwargs)

            res = []
            for p_name, p_type in self.params_config.items():
                if p_name in kwargs and not augmented_isinstance(kwargs[p_name], p_type):
                    res.append(p_name)

            if len(res) > 0:
                prefix_str = f"Function '{self.func_name}' parameter(s) type mismatch: "

                raise ParametersTypeError(prefix_str +
                                          ', '.join(
                                              [f"`{i}` only accept '{self.type2string(self.params_config[i])}' type"
                                               for i in res]) + '.')

            return func(**kwargs)

        return wrapper


class ParameterValuesAssert(BaseAssert):
    """
    function parameters values checker
    """

    def __init__(self, params_config, func_name=None):
        super().__init__(params_config, func_name)

    def check_params_config_type(self, params_config, func_name):
        super().check_params_config_type(params_config, func_name)

        for v in params_config.values():
            if augmented_isinstance(v, str) and not check_obj_is_function(eval(v)):
                raise ParametersValueError("If ParameterValuesAssert.params_config value is string, "
                                           "it must be a function of string-type wrapped. ")
            elif not augmented_isinstance(v, tuple) and not check_obj_is_function(eval(v)):
                raise ParametersValueError("If ParameterValuesAssert.params_config value is not string or tuple, "
                                           "it must be a callable function. ")

    @staticmethod
    def params_config_value_type(types=(tuple, str, FunctionType, LambdaType), names='(tuple, str, function)'):
        return types, names

    def __call__(self, func):
        self.func_name = self.func_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.check_func_params_exists(func)
            kwargs = get_function_all_kwargs(func, self.func_name, *args, **kwargs)

            res = []
            for p_name, p_values in self.params_config.items():
                if augmented_isinstance(p_values, tuple):
                    if p_name in kwargs and not kwargs[p_name] in p_values:
                        res.append(p_name)
                else:
                    if callable(eval(p_values)) and not eval(p_values)(kwargs[p_name]):
                        res.append(p_name)

            if len(res) > 0:
                prefix_str = f"Function '{self.func_name}' parameter(s) values mismatch: "

                raise ParametersValueError(prefix_str +
                                           ', '.join(
                                               [f"`{i}` must in or satisfy '{self.params_config[i]}' condition(s)"
                                                for i in res]) + '.')

            return func(**kwargs)

        return wrapper
