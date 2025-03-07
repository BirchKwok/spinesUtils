import pytest
import time


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="运行标记为慢的测试"
    )


def pytest_configure(config):
    """添加自定义标记"""
    config.addinivalue_line("markers", "slow: 标记为慢速执行的测试")


def pytest_collection_modifyitems(config, items):
    """根据命令行选项跳过测试"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="需要使用--run-slow选项才能运行")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def timer_precision():
    """测量当前系统的计时器精度"""
    measurements = []
    for _ in range(10):
        start = time.perf_counter()
        end = time.perf_counter()
        measurements.append(end - start)
    return min(measurements)  # 返回最小测量值作为计时器精度 