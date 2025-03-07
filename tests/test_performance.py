import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from spinesUtils.logging import Logger

def test_logging_performance():
    """测试日志记录性能"""
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        log_file = temp.name

    # 初始化日志记录器
    logger = Logger(
        name="PerformanceTest",
        fp=log_file,
        level="INFO",
        with_time=True,
        max_file_size=100 * 1024 * 1024,  # 100MB
        console_output=False,  # 关闭控制台输出以提高性能
        buffer_size=50000,  # 更大的缓冲区
        flush_interval=0.5  # 更频繁的刷新
    )

    # 测试参数
    num_threads = 20  # 增加线程数
    messages_per_thread = 50000  # 增加消息数量
    message = "This is a test message for performance testing" * 2  # ~100字节的消息

    def worker():
        """工作线程函数"""
        for i in range(messages_per_thread):
            logger.info(f"{message} - {i}")

    # 记录开始时间
    start_time = time.time()

    # 使用线程池并发写入日志
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for future in futures:
            future.result()

    # 确保所有消息都写入
    logger.flush()
    logger.close()

    # 计算总时间和性能指标
    end_time = time.time()
    total_time = end_time - start_time
    total_messages = num_threads * messages_per_thread
    messages_per_second = total_messages / total_time

    # 获取日志文件大小
    file_size = os.path.getsize(log_file)
    mb_per_second = (file_size / 1024 / 1024) / total_time

    # 清理临时文件
    os.unlink(log_file)

    # 打印性能指标
    print(f"\n性能测试结果:")
    print(f"总消息数: {total_messages:,}")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"每秒消息数: {messages_per_second:,.2f}")
    print(f"每秒写入: {mb_per_second:.2f} MB")
    print(f"平均每条消息大小: {file_size/total_messages:.2f} 字节")
    print(f"线程数: {num_threads}")

if __name__ == "__main__":
    test_logging_performance() 