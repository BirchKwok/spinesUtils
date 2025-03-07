import time
from spinesUtils.asserts import raise_if


class Timer:
    """
    A timer class for measuring elapsed time with additional support for pausing,
    resuming, and recording intermediate time points. It offers a context manager
    for easy use within a `with` block.

    Methods
    -------
    start():
        Starts the timer. Resets any previous timing.

    pause():
        Pauses the timer. Pausing again has no effect until resumed.

    resume():
        Resumes the timer if it was paused. Otherwise, has no effect.

    middle_point():
        Records an intermediate time point.

    end():
        Stops the timer and returns the total elapsed time. Resets the timer.

    last_timestamp_diff():
        Returns the time elapsed since the last recorded time point.

    total_elapsed_time():
        Returns the total time elapsed from the start, including pauses.

    get_elapsed_time():
        Returns the current elapsed time without stopping the timer.

    clear():
        Resets the timer and clears all recorded time points.

    sleep(secs):
        Sleeps for the given number of seconds and records the time point after waking up.

    is_running():
        Returns whether the timer is currently running (not stopped or paused).

    is_paused():
        Returns whether the timer is currently paused.

    is_started():
        Returns whether the timer has been started.

    get_middle_points():
        Returns a list of all recorded middle points.

    Examples
    --------
    >>> timer = Timer()
    ... with timer.session():
    ...     timer.sleep(1)
    ...     timer.middle_point()
    ...     timer.sleep(2)
    ...
    >>> t = Timer()
    >>> with t.session():
    ...    t.sleep(1)
    ...    print("Last step elapsed time:", round(t.last_timestamp_diff(), 2), 'seconds')
    ...    t.middle_point()
    ...    t.sleep(2)
    ...    print("Last step elapsed time:", round(t.last_timestamp_diff(), 2), 'seconds')
    ...    total_elapsed_time = t.total_elapsed_time()

    >>> print("Total Time:", round(total_elapsed_time, 2), 'seconds')
    Total Time: 3.0 seconds

    >>> timer = Timer()
    >>> timer.start()
    >>> timer.sleep(1)
    >>> timer.pause()
    >>> # Perform some operations during pause
    >>> timer.resume()
    >>> timer.sleep(2)
    >>> total_time = timer.end()
    >>> print("Total Time with Pause:", total_time)
    Total Time with Pause: 3.0
    """
    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._middle_points = []
        self._is_paused = False
        self._pause_time = None
        self._pause_duration = 0  # 累计暂停时间
        self._last_end_time = None

    def check_is_started(self):
        raise_if(RuntimeError, self._start_time is None, "Timer is not started.")

    def is_started(self):
        """Returns whether the timer has been started."""
        return self._start_time is not None

    def is_running(self):
        """Returns whether the timer is currently running (not stopped or paused)."""
        return self._start_time is not None and not self._is_paused

    def is_paused(self):
        """Returns whether the timer is currently paused."""
        return self._is_paused

    def start(self):
        """Starts the timer. Resets any previous timing."""
        self._start_time = time.perf_counter()  # 使用perf_counter获取更高精度
        self._end_time = None
        self._middle_points = []
        self._is_paused = False
        self._pause_duration = 0
        return self

    def pause(self):
        """Pauses the timer. Pausing again has no effect until resumed."""
        self.check_is_started()
        if not self._is_paused:
            self._pause_time = time.perf_counter()
            self._is_paused = True
        return self

    def resume(self):
        """Resumes the timer if it was paused. Otherwise, has no effect."""
        if self._is_paused:
            now = time.perf_counter()
            pause_duration = now - self._pause_time
            self._pause_duration += pause_duration
            self._is_paused = False
        return self

    def middle_point(self):
        """Records an intermediate time point."""
        self.check_is_started()
        if not self._is_paused:
            self._middle_points.append(time.perf_counter())
            return self
        else:
            # 在暂停状态下调用时给出警告
            print("警告: 计时器当前已暂停，无法记录中间点")
            return False

    def end(self):
        """Stops the timer and returns the total elapsed time. Resets the timer."""
        if self._start_time is not None:
            if self._is_paused:
                # 如果在暂停状态下结束，使用暂停时间作为结束时间
                elapsed_time = self._pause_time - self._start_time - self._pause_duration
                self._is_paused = False
            else:
                self._end_time = time.perf_counter()
                elapsed_time = self._end_time - self._start_time - self._pause_duration
            
            self._last_end_time = elapsed_time  # 保存最后一次计时结果
            self._start_time = None
            self._middle_points = []
            return elapsed_time
        return 0

    def get_elapsed_time(self):
        """Returns the current elapsed time without stopping the timer."""
        if not self.is_started():
            return 0
            
        if self._is_paused:
            return self._pause_time - self._start_time - self._pause_duration
        else:
            return time.perf_counter() - self._start_time - self._pause_duration

    def last_timestamp_diff(self):
        """Returns the time elapsed since the last recorded time point."""
        self.check_is_started()

        if self._is_paused:
            latest_point = self._middle_points[-1] if self._middle_points else self._start_time
            return self._pause_time - latest_point
        else:
            latest_point = self._middle_points[-1] if self._middle_points else self._start_time
            return time.perf_counter() - latest_point

    def total_elapsed_time(self):
        """Returns the total time elapsed from the start, including pauses."""
        return self.get_elapsed_time()

    def get_middle_points(self):
        """Returns a list of all recorded middle points relative to start time."""
        if not self.is_started():
            return []
            
        return [point - self._start_time for point in self._middle_points]

    def clear(self):
        """Resets the timer and clears all recorded time points."""
        self._start_time = None
        self._end_time = None
        self._middle_points = []
        self._is_paused = False
        self._pause_duration = 0
        return self

    def sleep(self, secs):
        """
        Sleeps for the given number of seconds and records the time point after waking up.
        
        Parameters
        ----------
        secs : float
            Number of seconds to sleep. Must be non-negative.
        """
        self.check_is_started()
        
        # 验证参数
        raise_if(ValueError, secs < 0, "Sleep duration must be non-negative.")
        
        if not self._is_paused:
            time.sleep(secs)
            self.middle_point()
        else:
            print("警告: 计时器当前已暂停，sleep方法不会记录中间点")
            time.sleep(secs)
        return self

    def session(self):
        """返回一个计时器会话上下文管理器"""
        return TimerSession(self)


class TimerSession:
    def __init__(self, timer=None):
        self._timer = timer or Timer()

    def __enter__(self):
        return self._timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.end()
