#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from spinesUtils.logging import FastLogger

# 测试参数
NUM_MESSAGES = 1_000_000  # 总消息数
NUM_THREADS = 20  # 线程数
MESSAGE = "这是一条测试日志消息，包含一些数据: value=12345, status='OK'"  # 测试消息


def setup_standard_logger():
    """设置标准Python logging库"""
    log_file = tempfile.NamedTemporaryFile(delete=False).name
    logger = logging.getLogger("standard_logger")
    logger.setLevel(logging.INFO)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


def setup_fast_logger():
    """设置FastLogger"""
    log_file = tempfile.NamedTemporaryFile(delete=False).name
    logger = FastLogger(
        name="fast_logger",
        fp=log_file,
        level="INFO",
        truncate_file=True,
        console_output=False,
        buffer_size=100000,
        flush_interval=1.0
    )
    
    return logger, log_file


def worker_standard_logger(logger, num_messages_per_thread):
    """标准logger的工作线程"""
    for _ in range(num_messages_per_thread):
        logger.info(MESSAGE)


def worker_fast_logger(logger, num_messages_per_thread):
    """FastLogger的工作线程"""
    for _ in range(num_messages_per_thread):
        logger.info(MESSAGE)


def test_standard_logger():
    """测试标准Python logging库的性能"""
    logger, log_file = setup_standard_logger()
    
    messages_per_thread = NUM_MESSAGES // NUM_THREADS
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for _ in range(NUM_THREADS):
            executor.submit(worker_standard_logger, logger, messages_per_thread)
    
    # 确保所有日志都写入
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    
    end_time = time.time()
    
    # 计算性能指标
    total_time = end_time - start_time
    messages_per_second = NUM_MESSAGES / total_time
    
    # 获取文件大小
    file_size = os.path.getsize(log_file)
    mb_per_second = (file_size / 1024 / 1024) / total_time
    bytes_per_message = file_size / NUM_MESSAGES
    
    # 清理
    os.unlink(log_file)
    
    return {
        "total_time": total_time,
        "messages_per_second": messages_per_second,
        "mb_per_second": mb_per_second,
        "bytes_per_message": bytes_per_message,
        "file_size": file_size
    }


def test_fast_logger():
    """测试FastLogger的性能"""
    logger, log_file = setup_fast_logger()
    
    messages_per_thread = NUM_MESSAGES // NUM_THREADS
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for _ in range(NUM_THREADS):
            executor.submit(worker_fast_logger, logger, messages_per_thread)
    
    # 确保所有日志都写入
    logger.flush()
    logger.close()
    
    end_time = time.time()
    
    # 计算性能指标
    total_time = end_time - start_time
    messages_per_second = NUM_MESSAGES / total_time
    
    # 获取文件大小
    file_size = os.path.getsize(log_file)
    mb_per_second = (file_size / 1024 / 1024) / total_time
    bytes_per_message = file_size / NUM_MESSAGES
    
    # 清理
    os.unlink(log_file)
    
    return {
        "total_time": total_time,
        "messages_per_second": messages_per_second,
        "mb_per_second": mb_per_second,
        "bytes_per_message": bytes_per_message,
        "file_size": file_size
    }


def format_number(num):
    """格式化数字，添加千位分隔符"""
    return f"{num:,.2f}"


def main():
    print(f"开始性能测试对比...")
    print(f"总消息数: {format_number(NUM_MESSAGES)}")
    print(f"线程数: {NUM_THREADS}")
    print(f"消息内容: '{MESSAGE}'")
    print("\n" + "="*50 + "\n")
    
    print("测试 Python 标准 logging 库...")
    std_results = test_standard_logger()
    
    print("\n测试 FastLogger...")
    fast_results = test_fast_logger()
    
    # 计算性能提升
    speedup = std_results["total_time"] / fast_results["total_time"]
    throughput_increase = fast_results["messages_per_second"] / std_results["messages_per_second"]
    
    # 打印结果表格
    print("\n" + "="*50)
    print("性能测试结果对比:")
    print("="*50)
    print(f"{'指标':<25} {'标准 logging':<20} {'FastLogger':<20} {'提升比例':<15}")
    print("-"*80)
    print(f"{'总时间 (秒)':<25} {format_number(std_results['total_time']):<20} {format_number(fast_results['total_time']):<20} {format_number(speedup)}x")
    print(f"{'每秒消息数':<25} {format_number(std_results['messages_per_second']):<20} {format_number(fast_results['messages_per_second']):<20} {format_number(throughput_increase)}x")
    print(f"{'每秒写入 (MB)':<25} {format_number(std_results['mb_per_second']):<20} {format_number(fast_results['mb_per_second']):<20} {format_number(fast_results['mb_per_second']/std_results['mb_per_second'])}x")
    print(f"{'平均每条消息大小 (字节)':<25} {format_number(std_results['bytes_per_message']):<20} {format_number(fast_results['bytes_per_message']):<20}")
    print(f"{'总文件大小 (MB)':<25} {format_number(std_results['file_size']/1024/1024):<20} {format_number(fast_results['file_size']/1024/1024):<20}")
    print("="*80)
    
    if speedup > 1:
        print(f"\n结论: FastLogger 比标准 logging 库快 {format_number(speedup)} 倍!")
    else:
        print(f"\n结论: 标准 logging 库比 FastLogger 快 {format_number(1/speedup)} 倍!")


if __name__ == "__main__":
    main() 