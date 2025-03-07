"""
性能测试运行脚本
用于测试断言装饰器的性能
"""
import sys
import os

# 将项目根目录添加到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_asserts import (
    test_parameter_type_assert_performance,
    test_parameter_values_assert_performance,
    test_nested_decorators_performance
)

def main():
    """运行所有断言装饰器的性能测试"""
    print("=" * 50)
    print("断言装饰器性能测试")
    print("=" * 50)
    
    print("\n1. 参数类型断言装饰器性能测试")
    test_parameter_type_assert_performance()
    
    print("\n2. 参数值断言装饰器性能测试")
    test_parameter_values_assert_performance()
    
    print("\n3. 嵌套断言装饰器性能测试")
    test_nested_decorators_performance()
    
    print("\n" + "=" * 50)
    print("性能测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main() 