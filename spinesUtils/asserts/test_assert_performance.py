import time
from typing import List, Dict, Optional, Any

from spinesUtils.asserts import (
    ParameterTypeAssert,
    ParameterValuesAssert
)


# 测试函数：类型验证
@ParameterTypeAssert({'a': int, 'b': str, 'c': list})
def function_with_type_assert(a, b, c):
    return a, b, c


# 使用类型提示的函数
@ParameterTypeAssert({})  # 空字典也会使用类型提示
def function_with_type_hints(a: int, b: str, c: List[int], d: Optional[Dict[str, Any]] = None):
    return a, b, c, d


# 测试函数：值验证
@ParameterValuesAssert({'a': 'lambda x: x > 0', 'b': ('x', 'y', 'z')})
def function_with_values_assert(a, b):
    return a, b


def test_functionality():
    """测试功能正确性"""
    print("功能测试：")
    
    # 测试类型验证
    result = function_with_type_assert(1, "test", [1, 2, 3])
    assert result == (1, "test", [1, 2, 3])
    print("类型验证测试通过")
    
    # 测试使用类型提示
    result = function_with_type_hints(1, "test", [1, 2, 3], {"key": "value"})
    assert result == (1, "test", [1, 2, 3], {"key": "value"})
    print("类型提示测试通过")
    
    # 测试值验证
    result = function_with_values_assert(5, "y")
    assert result == (5, "y")
    print("值验证测试通过")
    
    print("所有功能测试通过!\n")


def test_error_handling():
    """测试错误处理"""
    print("错误处理测试：")
    
    # 测试类型错误
    try:
        function_with_type_assert("not an int", "test", [1, 2, 3])
    except Exception as e:
        print(f"类型验证器错误: {type(e).__name__}: {e}")
    
    # 测试值错误
    try:
        function_with_values_assert(-5, "y")
    except Exception as e:
        print(f"值验证器错误 (范围): {type(e).__name__}: {e}")
    
    try:
        function_with_values_assert(5, "invalid")
    except Exception as e:
        print(f"值验证器错误 (枚举): {type(e).__name__}: {e}")
    
    print("错误处理测试完成!\n")


def benchmark(func, *args, **kwargs):
    """简单的性能测试函数"""
    iterations = 100000
    start_time = time.time()
    
    for _ in range(iterations):
        func(*args, **kwargs)
    
    end_time = time.time()
    return end_time - start_time


def test_performance():
    """测试性能"""
    print("性能测试 (100,000次调用)：")
    
    # 测试不带装饰器的函数性能
    def plain_function(a, b, c):
        return a, b, c
    
    plain_time = benchmark(plain_function, 1, "test", [1, 2, 3])
    
    # 类型验证性能测试
    type_assert_time = benchmark(function_with_type_assert, 1, "test", [1, 2, 3])
    
    # 使用类型提示的函数性能测试
    type_hints_time = benchmark(function_with_type_hints, 1, "test", [1, 2, 3], {"key": "value"})
    
    # 值验证性能测试
    values_assert_time = benchmark(function_with_values_assert, 5, "y")
    
    print(f"无装饰器函数耗时: {plain_time:.4f}秒")
    print(f"类型验证耗时: {type_assert_time:.4f}秒 (Pydantic优化版)")
    print(f"类型提示耗时: {type_hints_time:.4f}秒 (Pydantic优化版)")
    print(f"值验证耗时: {values_assert_time:.4f}秒 (Pydantic优化版)")
    print(f"类型验证装饰器开销比例: {type_assert_time / plain_time:.2f}倍")
    print(f"类型提示装饰器开销比例: {type_hints_time / plain_time:.2f}倍")
    print(f"值验证装饰器开销比例: {values_assert_time / plain_time:.2f}倍")


def test_cache_effect():
    """测试缓存效果"""
    print("\n缓存效果测试 (5次10,000次调用)：")
    
    print("测试类型验证函数:")
    for i in range(5):
        time_taken = benchmark(function_with_type_assert, 1, "test", [1, 2, 3]) / 10  # 降低迭代次数
        print(f"  - 第{i+1}次运行: {time_taken:.4f}秒")
    
    print("\n测试值验证函数:")
    for i in range(5):
        time_taken = benchmark(function_with_values_assert, 5, "y") / 10  # 降低迭代次数
        print(f"  - 第{i+1}次运行: {time_taken:.4f}秒")


if __name__ == "__main__":
    print("=" * 50)
    print("参数验证装饰器测试 (Pydantic优化版)")
    print("=" * 50)
    
    test_functionality()
    test_error_handling()
    test_performance()
    test_cache_effect() 