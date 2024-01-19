import os
import sys
from datetime import datetime

import pytz

from spinesUtils.asserts import ParameterTypeAssert
from spinesUtils.utils import get_env_variable


class Logger:
    """
    A customizable logger class for logging messages with different severity levels.
    The logger can output messages to stderr and optionally to a file.

    Parameters
    ----------
    name : None or str, optional
        The name of the logger. Default is None.
    fp : None or str, optional
        The file path where logs will be written. Default is None.
    level : str, optional (default='INFO')
        The logging level threshold. Messages below this level will not be logged.
    truncate_file : bool, optional (default=False)
        If True, truncates the file at 'fp' before starting logging.
    with_time : bool, optional (default=True)
        If True, includes timestamp in log messages.
    use_utc_time : bool, optional (default=False)
        If True, uses UTC time for timestamps.

    Methods
    -------
    log(msg, level='INFO'):
        Logs a message with the specified severity level.
    info(msg):
        Logs a message with 'INFO' severity level.
    error(msg):
        Logs a message with 'ERROR' severity level.
    debug(msg):
        Logs a message with 'DEBUG' severity level.
    critical(msg):
        Logs a message with 'CRITICAL' severity level.
    warning(msg):
        Logs a message with 'WARNING' severity level.

    Examples
    --------
    >>> logger = Logger(name="MyLogger", fp="log.txt", level="DEBUG")
    >>> logger.info("This is an info message.")
    >>> logger.error("This is an error message.")
    """
    _LOG_LEVELS = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
    _LOG_LEVELS_REVERSED = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}

    @ParameterTypeAssert({
        'name': (None, str), 'fp': (None, str),
        'level': str, 'truncate_file': bool,
        'with_time': bool, 'use_utc_time': bool
    }, func_name='Logger')
    def __init__(self, name=None, fp=None, level='INFO', truncate_file=False, with_time=True, use_utc_time=False):
        self.name = name
        self.fp = fp
        self.level = self._LOG_LEVELS.get(
            get_env_variable('SPS_LOG_LEVEL', default=level, default_type=str).upper(), 20)  # 默认为 INFO 级别
        self.with_time = with_time
        self.use_utc_time = use_utc_time

        self._file_handle = None
        if self.fp:
            if truncate_file:
                self._truncate_file()
            self._open_file()

    def __del__(self):
        self._close_file()

    def _truncate_file(self):
        try:
            if os.path.isfile(self.fp):
                with open(self.fp, 'w'):
                    pass
        except Exception as e:
            print(f"Error truncating file {self.fp}: {e}", file=sys.stderr)

    def _open_file(self):
        try:
            self._file_handle = open(self.fp, 'a')
        except Exception as e:
            print(f"Error opening file {self.fp}: {e}", file=sys.stderr)

    def _close_file(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _prefix_format(self):
        prefix = []
        if self.with_time:
            if self.use_utc_time:
                time_str = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            else:
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix.append(time_str)
        if self.name:
            prefix.append(self.name)
        if prefix:
            return ' - '.join(prefix)
        return ''

    @ParameterTypeAssert({'msg': str, 'level': str})
    def log(self, msg, level='INFO'):
        message_level = self._LOG_LEVELS.get(level.upper(), 20)
        if message_level < self.level:
            return

        prefix = self._prefix_format()
        if prefix:
            message = (self._prefix_format() + ' - ' + self._LOG_LEVELS_REVERSED[message_level] + ' - ' +
                       msg + '\n')
        else:
            message = ('[log] '+ self._LOG_LEVELS_REVERSED[message_level] + ' - ' + msg + '\n')

        # 如果文件句柄存在，则写入文件
        if self._file_handle:
            try:
                self._file_handle.write(message)
                self._file_handle.flush()
            except Exception as e:
                print(f"Error writing to file {self.fp}: {e}", file=sys.stderr)

        sys.stderr.write(message)

    @ParameterTypeAssert({'msg': str})
    def info(self, msg):
        return self.log(msg, "INFO")

    @ParameterTypeAssert({'msg': str})
    def error(self, msg):
        return self.log(msg, "ERROR")

    @ParameterTypeAssert({'msg': str})
    def debug(self, msg):
        return self.log(msg, "DEBUG")

    @ParameterTypeAssert({'msg': str})
    def critical(self, msg):
        return self.log(msg, "CRITICAL")

    @ParameterTypeAssert({'msg': str})
    def warning(self, msg):
        return self.log(msg, "WARNING")
