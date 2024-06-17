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
    log(msg, level='INFO', rewrite_print=False):
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
        self._last_message = ""  # 用于存储最后一条消息

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

    def _get_time_str(self):
        if self.use_utc_time:
            return datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _prefix_format(self, with_time=True):
        prefix = []
        if with_time:
            time_str = self._get_time_str()
            prefix.append(time_str)
        if self.name:
            prefix.append(self.name)
        if prefix:
            return ' - '.join(prefix)
        return ''

    @ParameterTypeAssert({'msg': str, 'level': str, 'rewrite_print': bool})
    def log(self, msg, level='INFO', rewrite_print=False):
        """
        Logs a message with the specified severity level.

        Parameters
        ----------
        msg : str
            The message to log.
        level : str, optional (default='INFO')
            The severity level of the message. Must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        rewrite_print : bool, optional (default=False)
            If True, rewrites the last message on the console.

        Returns
        -------
        None
        """
        message_level = self._LOG_LEVELS.get(level.upper(), 20)
        if message_level < self.level:
            return

        # 文件中始终带有时间戳
        file_prefix = self._prefix_format(with_time=True)
        if file_prefix:
            file_message = (file_prefix + ' - ' + self._LOG_LEVELS_REVERSED[message_level] + ' - ' + msg + '\n')
        else:
            file_message = ('[log] ' + self._LOG_LEVELS_REVERSED[message_level] + ' - ' + msg + '\n')

        # 控制台输出根据 with_time 参数决定是否带时间戳
        console_prefix = self._prefix_format(with_time=self.with_time)
        if console_prefix:
            console_message = (console_prefix + ' - ' + self._LOG_LEVELS_REVERSED[message_level] + ' - ' + msg + '\r')
        else:
            console_message = ('[log] ' + self._LOG_LEVELS_REVERSED[message_level] + ' - ' + msg + '\r')

        # 如果文件句柄存在，则写入文件
        if self._file_handle:
            try:
                self._file_handle.write(file_message)
                self._file_handle.flush()
            except Exception as e:
                print(f"Error writing to file {self.fp}: {e}", file=sys.stderr)

        # 处理控制台输出
        if rewrite_print:
            if msg != self._last_message:
                sys.stderr.write('\n' + console_message)
            else:
                sys.stderr.write('\r' + console_message)
            self._last_message = msg
        else:
            sys.stderr.write('\n' + console_message)
            self._last_message = msg

        sys.stderr.flush()

    @ParameterTypeAssert({'msg': str})
    def info(self, msg, rewrite_print=False):
        """
        Logs a message with 'INFO' severity level.

        Parameters
        ----------
        msg : str
            The message to log.
        rewrite_print : bool, optional (default=False)
            If True, rewrites the last message on the console.

        Returns
        -------
        None
        """
        return self.log(msg, "INFO", rewrite_print=rewrite_print)

    @ParameterTypeAssert({'msg': str})
    def error(self, msg, rewrite_print=False):
        """
        Logs a message with 'ERROR' severity level.

        Parameters
        ----------
        msg : str
            The message to log.
        rewrite_print : bool, optional (default=False)
            If True, rewrites the last message on the console.

        Returns
        -------
        None
        """
        return self.log(msg, "ERROR", rewrite_print=rewrite_print)

    @ParameterTypeAssert({'msg': str})
    def debug(self, msg, rewrite_print=False):
        """
        Logs a message with 'DEBUG' severity level.

        Parameters
        ----------
        msg : str
            The message to log.
        rewrite_print : bool, optional (default=False)
            If True, rewrites the last message on the console.

        Returns
        -------
        None
        """
        return self.log(msg, "DEBUG", rewrite_print=rewrite_print)

    @ParameterTypeAssert({'msg': str})
    def critical(self, msg, rewrite_print=False):
        """
        Logs a message with 'CRITICAL' severity level.

        Parameters
        ----------
        msg : str
            The message to log.
        rewrite_print : bool, optional (default=False)
            If True, rewrites the last message on the console.

        Returns
        -------
        None
        """
        return self.log(msg, "CRITICAL", rewrite_print=rewrite_print)

    @ParameterTypeAssert({'msg': str})
    def warning(self, msg, rewrite_print=False):
        """
        Logs a message with 'WARNING' severity level.

        Parameters
        ----------
        msg : str
            The message to log.
        rewrite_print : bool, optional (default=False)
            If True, rewrites the last message on the console.

        Returns
        -------
        None
        """
        return self.log(msg, "WARNING", rewrite_print=rewrite_print)
