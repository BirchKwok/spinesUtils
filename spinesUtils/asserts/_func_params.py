
def get_function_params(func):
    """get parameters of function"""
    return func.__code__.co_varnames[:func.__code__.co_argcount]


def generate_function_kwargs(func, args, kwargs=None):
    new_kwargs = {}

    # 获取位置参数
    func_params = get_function_params(func)

    for args_param, args_value in zip(func_params[:len(args)], args):
        new_kwargs[args_param] = args_value

    if func.__defaults__ is not None:
        # 获取默认参数
        for default_param, default_value in zip(func_params[::-1],
                                                func.__defaults__[::-1]):
            if default_param not in new_kwargs:
                new_kwargs[default_param] = default_value

    for k, v in kwargs.items():
        new_kwargs[k] = v

    return new_kwargs
