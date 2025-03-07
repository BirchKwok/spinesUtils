import pytest
import time
from typing import Union
from functools import wraps

from spinesUtils.asserts import (
    ParameterTypeAssert, 
    ParameterValuesAssert,
    ParametersTypeError,
    ParametersValueError
)


# 功能测试部分 - ParameterTypeAssert
def test_parameter_type_assert_valid():
    """测试参数类型断言装饰器 - 有效参数"""
    
    @ParameterTypeAssert({'x': int, 'y': (int, float)})
    def func(x, y=1):
        return x + y
    
    # 测试有效参数
    assert func(1, 2) == 3
    assert func(1, 2.5) == 3.5
    assert func(x=1, y=2) == 3
    assert func(1) == 2  # 使用默认值


def test_parameter_type_assert_invalid():
    """测试参数类型断言装饰器 - 无效参数"""
    
    @ParameterTypeAssert({'x': int, 'y': (int, float)})
    def func(x, y=1):
        return x + y
    
    # 测试无效参数
    with pytest.raises(ParametersTypeError):
        func("1", 2)  # x 应该是 int
    
    with pytest.raises(ParametersTypeError):
        func(1, "2")  # y 应该是 int 或 float
    
    with pytest.raises(ParametersTypeError):
        func(x="1", y=2)  # x 应该是 int


def test_parameter_type_assert_complex_types():
    """测试参数类型断言装饰器 - 复杂类型"""
    
    @ParameterTypeAssert({
        'a': list,
        'b': (dict, None),
        'c': Union[str, int],  # Union 类型会被转换为元组
    })
    def func(a, b=None, c="test"):
        return len(a) + (len(b) if b else 0) + (len(c) if isinstance(c, str) else c)
    
    # 测试有效参数
    assert func([1, 2, 3], {'key': 'value'}, 'abc') == 3 + 1 + 3
    assert func([1, 2], None, 5) == 2 + 0 + 5
    
    # 测试无效参数
    with pytest.raises(ParametersTypeError):
        func(a=1, b=None, c="test")  # a 应该是 list
    
    with pytest.raises(ParametersTypeError):
        func([1, 2], [], "test")  # b 应该是 dict 或 None


def test_parameter_type_assert_none_type():
    """测试参数类型断言装饰器 - None类型"""
    
    @ParameterTypeAssert({
        'x': (int, None),
        'y': None,
    })
    def func(x=None, y=None):
        return (x or 0) + (y or 0)
    
    # 测试有效参数
    assert func(1, None) == 1
    assert func(None, None) == 0
    
    # 测试无效参数
    with pytest.raises(ParametersTypeError):
        func(x="1")  # x 应该是 int 或 None
    
    with pytest.raises(ParametersTypeError):
        func(y=1)  # y 应该是 None


# 功能测试部分 - ParameterValuesAssert
def test_parameter_values_assert_valid():
    """测试参数值断言装饰器 - 有效参数"""
    
    @ParameterValuesAssert({
        'x': (1, 2, 3),  # 枚举值
        'y': lambda v: v > 0,  # Lambda 函数
    })
    def func(x, y):
        return x * y
    
    # 测试有效参数
    assert func(1, 2) == 2
    assert func(2, 3) == 6
    assert func(3, 0.5) == 1.5
    
    # 测试无效参数
    with pytest.raises(ParametersValueError):
        func(0, 1)  # x 应该在 (1, 2, 3) 中
    
    with pytest.raises(ParametersValueError):
        func(1, 0)  # y 应该大于 0
    
    with pytest.raises(ParametersValueError):
        func(1, -1)  # y 应该大于 0


def test_parameter_values_assert_string_func():
    """测试参数值断言装饰器 - 字符串函数"""
    
    @ParameterValuesAssert({
        'x': "lambda v: v > 0",  # 字符串包装的函数
        'y': "lambda v: isinstance(v, str) and len(v) > 2",  # 字符串包装的函数
    })
    def func(x, y):
        return str(x) + y
    
    # 测试有效参数
    assert func(1, "abc") == "1abc"
    assert func(10, "hello") == "10hello"
    
    # 测试无效参数
    with pytest.raises(ParametersValueError):
        func(0, "abc")  # x 应该大于 0
    
    with pytest.raises(ParametersValueError):
        func(1, "ab")  # y 应该是长度大于2的字符串


def test_parameter_values_assert_complex_conditions():
    """测试参数值断言装饰器 - 复杂条件"""
    
    def check_list(v):
        """检查列表元素是否都是正数"""
        return isinstance(v, list) and all(i > 0 for i in v)
    
    @ParameterValuesAssert({
        'a': check_list,  # 自定义函数
        'b': (None, 1, 2),  # 枚举值
        'c': "lambda v: isinstance(v, dict) and 'key' in v",  # 字符串包装的函数
    })
    def func(a, b=None, c=None):
        if c is None:
            c = {"key": "value"}
        result = sum(a) + (b or 0)
        return result, c.get("key")
    
    # 测试有效参数
    assert func([1, 2, 3], 1, {"key": "test"}) == (7, "test")  # 7 = sum([1,2,3]) + 1
    assert func([1, 2, 3], None) == (6, "value")  # 6 = sum([1,2,3]) + 0
    
    # 测试无效参数
    with pytest.raises(ParametersValueError):
        func([1, 0, -1], 1)  # a 应该是全正数列表
    
    with pytest.raises(ParametersValueError):
        func([1, 2, 3], 3)  # b 应该在 (None, 1, 2) 中
    
    with pytest.raises(ParametersValueError):
        func([1, 2, 3], 1, {})  # c 应该包含 'key'


# 性能测试部分 - 使用不同大小的输入和频繁调用测试性能
def benchmark_decorator(iterations=1000):
    """测量装饰器性能的基准测试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            for _ in range(iterations):
                result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} 执行 {iterations} 次用时: {end_time - start_time:.6f} 秒")
            return result
        return wrapper
    return decorator


def test_parameter_type_assert_performance():
    """测试参数类型断言装饰器的性能"""
    
    @benchmark_decorator(iterations=10000)
    @ParameterTypeAssert({
        'a': int,
        'b': (int, float),
        'c': (str, None),
        'd': list,
        'e': dict,
    })
    def complex_func(a, b=1.0, c=None, d=None, e=None):
        if d is None:
            d = []
        if e is None:
            e = {}
        return a + b + len(d) + len(e) + (len(c) if c else 0)
    
    # 运行一次以测量性能
    result = complex_func(100, 2.5, "test", [1, 2], {"a": 1})
    assert result == 100 + 2.5 + 2 + 1 + 4  # 2是列表长度，1是字典长度，4是字符串长度


def test_parameter_values_assert_performance():
    """测试参数值断言装饰器的性能"""
    
    @benchmark_decorator(iterations=10000)
    @ParameterValuesAssert({
        'a': (1, 2, 3, 4, 5),
        'b': lambda v: 0 <= v <= 10,
        'c': "lambda v: v is None or len(v) > 0",
        'd': lambda v: v is None or isinstance(v, list) and len(v) < 5,
        'e': "lambda v: v is None or isinstance(v, dict) and 'key' in v",
    })
    def complex_func(a, b=1.0, c=None, d=None, e=None):
        if d is None:
            d = []
        if e is None:
            e = {"key": "value"}
        return a + b + (len(c) if c else 0) + len(d) + len(e)
    
    # 运行一次以测量性能
    result = complex_func(3, 5, "test", [1, 2], {"key": "value"})
    assert result == 3 + 5 + 4 + 2 + 1


def test_nested_decorators_performance():
    """测试嵌套断言装饰器的性能"""
    
    @benchmark_decorator(iterations=5000)
    @ParameterTypeAssert({
        'a': int,
        'b': (float, int),
        'c': (str, None),
    })
    @ParameterValuesAssert({
        'a': lambda v: v > 0,
        'b': lambda v: v >= 0,
        'c': lambda v: v is None or len(v) > 2,
    })
    def nested_func(a, b=0, c=None):
        return a + b + (len(c) if c else 0)
    
    # 运行一次以测量性能
    result = nested_func(5, 2.5, "hello")
    assert result == 5 + 2.5 + 5


def test_compare_original_vs_optimized():
    """比较原始实现与优化版本的性能差异
    
    注：由于我们已经优化了代码，此测试仅作为未来比较的参考。
    实际比较时需要使用原始版本和优化版本分别测试。
    """
    # 这个测试函数是一个占位符，实际测试需要有原始版本的断言类
    
    @benchmark_decorator(iterations=10000)
    @ParameterTypeAssert({
        'x': int,
        'y': (int, float, str),
    })
    def optimized_func(x, y=1):
        return x + (float(y) if isinstance(y, str) else y)
    
    # 运行优化版本
    result = optimized_func(1, 2.5)
    assert result == 3.5
    
    # 在实际比较中，这里应该运行原始版本
    # 然后比较两者的性能差异
    print("优化版本与原始版本的性能比较需要分别运行两个版本，此处仅为参考")


if __name__ == "__main__":
    # 执行单元测试
    pytest.main(["-v", "test_asserts.py"]) 