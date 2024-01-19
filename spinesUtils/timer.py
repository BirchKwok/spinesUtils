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

    clear():
        Resets the timer and clears all recorded time points.

    sleep(secs):
        Sleeps for the given number of seconds and records the time point after waking up.

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
        self._last_end_time = None

    def check_is_started(self):
        raise_if(RuntimeError, self._start_time is None, "Timer is not started.")

    def start(self):
        """Starts the timer. Resets any previous timing."""
        self._start_time = time.time()
        self._end_time = None
        self._middle_points = []
        return self

    def pause(self):
        """Pauses the timer. Pausing again has no effect until resumed."""
        self.check_is_started()
        if not self._is_paused:
            self._pause_time = time.time()
            self._is_paused = True
        return self

    def resume(self):
        """Resumes the timer if it was paused. Otherwise, has no effect."""
        if self._is_paused:
            pause_duration = time.time() - self._pause_time
            self._start_time += pause_duration
            self._middle_points = [mp + pause_duration for mp in self._middle_points]
            self._is_paused = False
        return self

    def middle_point(self):
        """Records an intermediate time point."""
        self.check_is_started()
        if not self._is_paused:
            self._middle_points.append(time.time())
        return self

    def end(self):
        """Stops the timer and returns the total elapsed time. Resets the timer."""
        if self._start_time is not None and not self._is_paused:
            self._end_time = time.time()
            time_diff = self._end_time - self._start_time
            self._start_time = None
            self._middle_points = []
            return time_diff
        return 0

    def last_timestamp_diff(self):
        """Returns the time elapsed since the last recorded time point."""
        self.check_is_started()

        latest_point = self._middle_points[-1] if self._middle_points else self._start_time
        return time.time() - latest_point

    def total_elapsed_time(self):
        """Returns the total time elapsed from the start, including pauses."""
        if self._start_time is not None:
            return (self._end_time if self._end_time else time.time()) - self._start_time
        return 0

    def clear(self):
        """Resets the timer and clears all recorded time points."""
        self._start_time = None
        self._end_time = None
        self._middle_points = []
        self._is_paused = False
        return self

    def sleep(self, secs):
        """Sleeps for the given number of seconds and records the time point after waking up."""
        self.check_is_started()

        time.sleep(secs)
        return self

    def session(self):
        return TimerSession(self)


class TimerSession:
    def __init__(self, timer=None):
        self._timer = timer

    def __enter__(self):
        return self._timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.end()
