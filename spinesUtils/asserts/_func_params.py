import inspect
from inspect import signature


def get_function_params_name(func):
    """get parameters of function"""
    return list(signature(func).parameters.keys())


def generate_function_kwargs(func, *args, **kwargs):
    new_kwargs = {}

    # 获取位置参数
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


def check_has_params(func, params):
    """检查函数是否有指定形参"""
    sig = inspect.signature(func)
    param = sig.parameters.get(params, None)
    if param is not None:
        return True
    else:
        return False
