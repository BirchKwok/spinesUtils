from functools import wraps
from typing import Union

from ._type_and_exceptions import ParametersTypeError, augmented_isinstance, NotFittedError
from ._func_params import get_function_params, generate_function_kwargs


class BaseAssert:
    def __init__(self, params_config, func_name=None):
        self.check_type_assert_params(params_config, func_name)
        self.params_config = params_config
        self.func_name = func_name

    @staticmethod
    def check_type_assert_params(params_config, func_name):
        if not augmented_isinstance(func_name, (None, str)):
            raise ParametersTypeError("Parameter `func_name` of Function `TypeAssert` must be str or None.")

        if not augmented_isinstance(params_config, dict):
            raise ParametersTypeError("Parameter `params_config` of Function `TypeAssert` must be dict.")
        for k, v in params_config.items():
            if not augmented_isinstance(k, str):
                raise ParametersTypeError("Keys of params_config of Function `TypeAssert` must be string type.")

    def raise_params_numbers_error(self, func):
        raise ParametersTypeError(f"Function {self.func_name or func.__name__} only "
                                  f"accept {func.__code__.co_argcount} parameter(s)")

    def __call__(self, func):
        raise NotImplementedError


class TypeAssert(BaseAssert):
    """
    function parameters type checker

    """

    def __init__(self, params_config, func_name=None):
        super().__init__(params_config, func_name)

    @staticmethod
    def check_type_assert_params(params_config, func_name):
        if not augmented_isinstance(func_name, (None, str)):
            raise ParametersTypeError("Parameter `func_name` of Function `TypeAssert` must be str or None.")

        if not augmented_isinstance(params_config, dict):
            raise ParametersTypeError("Parameter `params_config` of Function `TypeAssert` must be dict.")
        for k, v in params_config.items():
            if not augmented_isinstance(k, str):
                raise ParametersTypeError("Keys of params_config of Function `TypeAssert` must be string type.")

            if not augmented_isinstance(v, (tuple, type, None)):
                raise ParametersTypeError("Values of params_config of Function `TypeAssert` must be tuple of class"
                                          " or class or tuple of None.")

    @staticmethod
    def type2string(tp: Union[type, tuple]):
        if augmented_isinstance(tp, type):
            return tp.__name__ if tp is not None else 'None'
        else:
            return [i.__name__ if i is not None else 'None' for i in tp]

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            input_size = len(args) + len(kwargs)

            kwargs = generate_function_kwargs(func, args, kwargs)
            if input_size > len(kwargs):
                self.raise_params_numbers_error(func)

            res = []

            for p_name, p_type in self.params_config.items():
                if p_name not in get_function_params(func):
                    raise ParametersTypeError(
                        f"Function '{self.func_name or func.__name__}' not exists `{p_name}` parameter.")

                if p_name in kwargs and not augmented_isinstance(kwargs[p_name], p_type):
                    res.append(p_name)

            if len(res) > 0:
                prefix_str = f"Function '{self.func_name or func.__name__}' parameter(s) type mismatch: "

                raise ParametersTypeError(prefix_str +
                                          ', '.join(
                                              [f"`{i}` only accept '{self.type2string(self.params_config[i])}' type"
                                               for i in res]) + '.')

            return func(**kwargs)

        return wrapper
