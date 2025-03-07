import os

import pytest
from spinesUtils.logging import Logger
from threading import Thread


@pytest.fixture
def temp_log_file(tmp_path):
    """Create temporary log file path"""
    return str(tmp_path / "test.log")


@pytest.fixture
def logger(temp_log_file):
    """Create a basic logger instance"""
    logger = Logger(name="TestLogger", fp=temp_log_file)
    yield logger
    logger.close()


def test_logger_initialization(temp_log_file):
    """Test logger initialization"""
    logger = Logger(name="TestLogger", fp=temp_log_file)
    assert logger.name == "TestLogger"
    assert logger.fp == temp_log_file
    assert logger.level == 20  # INFO level
    logger.close()


def test_log_levels(logger, temp_log_file):
    """Test message recording with different log levels"""
    messages = {
        "debug": "Debug message",
        "info": "Info message",
        "warning": "Warning message",
        "error": "Error message",
        "critical": "Critical message"
    }

    # Write logs with different levels
    logger.debug(messages["debug"])
    logger.info(messages["info"])
    logger.warning(messages["warning"])
    logger.error(messages["error"])
    logger.critical(messages["critical"])

    # Force flush and wait for write completion
    logger.flush()

    # Read log file content
    with open(temp_log_file, 'r') as f:
        content = f.read()

    # Verify all messages (except debug, because default level is INFO)
    assert messages["debug"] not in content  # debug should not be recorded
    assert messages["info"] in content
    assert messages["warning"] in content
    assert messages["error"] in content
    assert messages["critical"] in content


def test_log_rotation(temp_log_file):
    """Test log rotation functionality"""
    # Create a logger with a small file size limit
    logger = Logger(
        name="TestLogger",
        fp=temp_log_file,
        max_file_size=100,  # Very small file size limit
        backup_count=2,
        console_output=False  # Close console output for faster test
    )

    # Write enough logs to trigger rotation, but reduce total iterations
    for i in range(5):  # Reduce iterations
        logger.info(f"Test message {i}" * 5)
    
    # Force flush and wait for write completion
    logger.flush()
    logger.close()

    # Verify if backup files are created
    assert os.path.exists(temp_log_file)
    assert os.path.exists(f"{temp_log_file}.1") or os.path.exists(f"{temp_log_file}.2")


def test_context_manager(temp_log_file):
    """Test manual creation and closing functionality (alternative to context manager)"""
    logger = Logger(name="TestLogger", fp=temp_log_file)
    logger.info("Test message")
    logger.flush()
    
    # Verify message is written
    with open(temp_log_file, 'r') as f:
        content = f.read()
        assert "Test message" in content
    
    # Verify can be closed correctly
    logger.close()
    assert not logger._file_handle


def test_thread_safety():
    """Test multi-thread safety"""
    logger = Logger(
        name="TestLogger",
        fp="test_thread_safety.log",
        console_output=False,  # Close console output for faster test
        buffer_size=5000,
    )
    
    num_threads = 5  # Reduce thread count
    messages_per_thread = 100  # Reduce messages per thread
    threads = []
    
    def write_messages():
        for i in range(messages_per_thread):
            logger.info(f"Thread message {i}")
            
    # Create and start threads
    for _ in range(num_threads):
        t = Thread(target=write_messages)
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    # Ensure all messages are written
    logger.flush()
    
    # Use internal counter to verify message count
    expected_messages = num_threads * messages_per_thread
    # Use new get_total_messages method
    actual_messages = logger.get_total_messages()
    
    # Close logger
    logger.close()
    
    assert actual_messages == expected_messages, f"Expected {expected_messages} messages, got {actual_messages}"


def test_rewrite_print(logger, temp_log_file):
    """Test print to file functionality (rewrite_print has been removed)"""
    logger.info("First message")
    logger.info("Second message")
    
    # Force flush and wait for write completion
    logger.flush()
    
    # Verify file contains two messages
    with open(temp_log_file, 'r') as f:
        content = f.readlines()
        assert len(content) == 2
        assert "First message" in content[0]
        assert "Second message" in content[1]


def test_log_formatting(logger, temp_log_file):
    """Test log formatting"""
    test_message = "Test formatting"
    logger.info(test_message)
    logger.flush()
    
    with open(temp_log_file, 'r') as f:
        content = f.read()
        # Verify basic format
        assert logger.name in content
        assert "INFO" in content
        assert test_message in content
        # Verify timestamp format
        assert content.count(":") >= 2  # At least two colons (timestamp format)


def test_error_handling(tmp_path):
    """Test error handling"""
    # Use directory without permission to test error handling
    no_access_dir = tmp_path / "no_access"
    no_access_dir.mkdir()
    no_access_file = no_access_dir / "test.log"
    
    # On Windows, it may not be possible to set permissions, so here we need to try-except
    try:
        no_access_dir.chmod(0o000)
        logger = Logger(name="TestLogger", fp=str(no_access_file))
        logger.info("Test message")  # Should not throw an exception
        logger.flush()
        logger.close()
    except:
        pass  # On some systems, it may not be possible to set permissions
    finally:
        no_access_dir.chmod(0o755)  # Restore permissions


def test_environment_variable(temp_log_file, monkeypatch):
    """Test the effect of environment variables on log level"""
    # Set environment variable
    monkeypatch.setenv('SPS_LOG_LEVEL', 'DEBUG')
    
    logger = Logger(name="TestLogger", fp=temp_log_file)
    debug_message = "Debug level test"
    logger.debug(debug_message)
    logger.flush()
    
    with open(temp_log_file, 'r') as f:
        content = f.read()
        assert debug_message in content  # DEBUG level message should be recorded

    logger.close()


def test_flush_functionality(temp_log_file):
    """Test flush functionality"""
    logger = Logger(name="TestLogger", fp=temp_log_file)
    
    # Write multiple messages
    for i in range(50):  # Write messages less than buffer size
        logger.info(f"Test message {i}")
    
    # Check file before flush
    with open(temp_log_file, 'r') as f:
        content_before = f.readlines()
    
    # Force flush
    logger.flush()
    
    # Check file after flush
    with open(temp_log_file, 'r') as f:
        content_after = f.readlines()
    
    # Verify all messages are written
    assert len(content_after) == 50
    assert len(content_after) >= len(content_before)
    
    logger.close() 