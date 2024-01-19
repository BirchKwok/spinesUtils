import inspect
import re
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
    """
    Collects all keyword arguments for a function, including defaults, and checks if the number of arguments passed
    is not greater than the function accepts.

    Parameters
    ----------
    func : callable
        The function to collect keyword arguments for.
    func_name : str
        The name of the function, used for error messages.
    *args : Positional arguments to pass to the function.
    **kwargs : Keyword arguments to pass to the function.

    Returns
    -------
    dict
        A dictionary of all arguments that would be passed to the function.

    Raises
    ------
    ParametersTypeError
        If the number of arguments provided is greater than the function accepts.

    Examples
    --------
    >>> def sample_function(x, y=1):
    ...     return x + y
    >>> get_function_all_kwargs(sample_function, 'sample_function', 2)
    {'x': 2, 'y': 1}
    """
    input_size = len(args) + len(kwargs)

    kwargs = generate_function_kwargs(func, *args, **kwargs)

    if input_size > len(kwargs):
        raise_params_numbers_error(func, func_name=func_name or func.__name__)

    return kwargs


def check_obj_is_function(obj):
    """
    Checks if an object is a function or a lambda.

    Parameters
    ----------
    obj : any
        The object to check.

    Returns
    -------
    bool
        True if the object is a function or lambda, False otherwise.

    Examples
    --------
    >>> check_obj_is_function(lambda x: x)
    True
    >>> check_obj_is_function(42)
    False
    """
    if not augmented_isinstance(obj, (FunctionType, LambdaType)):
        return False
    return True


class BaseAssert:
    """
    Base class for assertion decorators used for function parameter validation.

    Attributes
    ----------
    cached_params : dict
        Cache of parameters names for functions.
    params_config : dict
        Configuration for parameters validation.

    Methods
    -------
    cache_func_params(func):
        Cache the parameter names of the function.
    check_func_params_exists(func):
        Check if the function has all the parameters specified in the configuration.
    check_params_config_type(params_config, func_name):
        Check the type of the params_config dictionary and its contents.

    Examples
    --------
    >>> @BaseAssert(params_config={'x': int, 'y': int}, func_name='sample_function')
    ... def sample_function(x, y=1):
    ...     return x + y
    """
    def __init__(self, params_config, func_name=None):
        # 现在，在构造函数中预先获取并缓存函数参数
        self.cached_params = {}
        self._params_config_value_types, self._params_config_value_type_names = self.params_config_value_type()
        self.check_params_config_type(params_config, func_name)
        self.params_config = params_config
        self.func_name = func_name

    def cache_func_params(self, func):
        # 缓存函数参数
        if func not in self.cached_params:
            self.cached_params[func] = get_function_params_name(func)

    def check_func_params_exists(self, func):
        # 使用缓存的参数进行检查
        self.cache_func_params(func)
        for p_name in self.params_config:
            if p_name not in self.cached_params[func]:
                raise ParametersTypeError(
                    f"Function '{self.func_name}' not exists `{p_name}` parameter.")

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

    def __call__(self, func):
        raise NotImplementedError


class ParameterTypeAssert(BaseAssert):
    """
    A decorator to assert that the parameters passed to a function match the expected types.

    Inherits from BaseAssert.

    Examples
    --------
    >>> @ParameterTypeAssert({'x': int, 'y': (int, type(None))})
    ... def sample_function(x, y=None):
    ...     return x if y is None else x + y
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
        self.cache_func_params(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.check_func_params_exists(func)
            kwargs = get_function_all_kwargs(func, self.func_name, *args, **kwargs)

            # 优化的类型检查逻辑
            mismatched_params = [
                p_name for p_name, p_type in self.params_config.items()
                if p_name in kwargs and not augmented_isinstance(kwargs[p_name], p_type)
            ]

            if mismatched_params:
                error_msg = self.build_type_error_msg(mismatched_params)
                raise ParametersTypeError(error_msg)

            return func(**kwargs)

        return wrapper

    def build_type_error_msg(self, mismatched_params):
        # 分离出错误信息构建逻辑
        prefix_str = f"Function '{self.func_name}' parameter(s) type mismatch: "

        return prefix_str + ', '.join(
            [f"{p} only accept '{self.type2string(self.params_config[p])}' type"
             for p in mismatched_params]) + '.'


class ParameterValuesAssert(BaseAssert):
    """
    A decorator to assert that the parameters passed to a function match the expected values or conditions.

    Inherits from BaseAssert.

    Examples
    --------
    >>> @ParameterValuesAssert({'x': lambda val: val > 0})
    ... def sample_function(x):
    ...     return x
    """

    def __init__(self, params_config, func_name=None):
        super().__init__(params_config, func_name)
        self.name_cache = {}  # 添加缓存
        self.pattern = re.compile(r"@ParameterValuesAssert\((\{.*?\})\)", re.DOTALL)

    def check_params_config_type(self, params_config, func_name):
        super().check_params_config_type(params_config, func_name)

        for v in params_config.values():
            if augmented_isinstance(v, str) and not check_obj_is_function(eval(v)):
                raise ParametersValueError("If ParameterValuesAssert.params_config value is string, "
                                           "it must be a function of string-type wrapped. ")
            elif not augmented_isinstance(v, (tuple, str)) and not check_obj_is_function(v):
                raise ParametersValueError("If ParameterValuesAssert.params_config value is not string or tuple, "
                                           "it must be a callable function. ")

    @staticmethod
    def params_config_value_type(types=(tuple, str, FunctionType, LambdaType), names='(tuple, str, function)'):
        return types, names

    def __call__(self, func):
        self.func_name = self.func_name or func.__name__
        self.cache_func_params(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.check_func_params_exists(func)
            kwargs = get_function_all_kwargs(func, self.func_name, *args, **kwargs)

            # 优化的值检查逻辑
            mismatched_params = self.get_mismatched_value_params(kwargs)

            if mismatched_params:
                error_msg = self.build_values_error_msg(mismatched_params)
                raise ParametersValueError(error_msg)

            return func(**kwargs)

        return wrapper

    def get_mismatched_value_params(self, kwargs):
        mismatched_params = []
        for p_name, p_values in self.params_config.items():
            if not self.is_value_valid(p_name, p_values, kwargs):
                mismatched_params.append(p_name)
        return mismatched_params

    @staticmethod
    def is_value_valid(p_name, p_values, kwargs):
        if augmented_isinstance(p_values, tuple):
            return kwargs.get(p_name) in p_values
        elif augmented_isinstance(p_values, str):
            return eval(p_values)(kwargs.get(p_name))
        else:
            return p_values(kwargs.get(p_name))

    def build_values_error_msg(self, mismatched_params):
        # 分离出错误信息构建逻辑
        prefix_str = f"Function '{self.func_name}' parameter(s) values mismatch: "
        return prefix_str + ', '.join(
            [f"`{p}` must in or satisfy '{self.get_name(self.params_config[p], p)}' condition(s)"
             for p in mismatched_params]) + '.'

    def get_name(self, p_values, func):
        # 使用缓存
        if func in self.name_cache:
            return self.name_cache[func]

        if augmented_isinstance(p_values, str):
            result = p_values
        elif callable(p_values):
            source = inspect.getsource(p_values).strip()
            match = self.pattern.search(source)
            result = match.group(1) if match else source
        else:
            result = str(p_values)

        self.name_cache[func] = result  # 存储结果到缓存
        return result
