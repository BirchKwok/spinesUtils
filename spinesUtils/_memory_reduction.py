import ctypes
from functools import wraps
import sys
from os import getpid

from psutil import Process


if sys.platform == 'darwin':
    malloc_trim = ctypes.CDLL(ctypes.util.find_library('System')).malloc_zone_pressure_relief
else:
    malloc_trim = ctypes.CDLL('libc.so.6').malloc_trim


def memory_reduction(func):
    """Reclaim the unused fragmented memory."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        malloc_trim(0)
        return res
    return wrapper


def GetMemUsage():
    """
    This function defines the memory usage across the kernel.
    Source-
    https://stackoverflow.com/questions/61366458/how-to-find-memory-usage-of-kaggle-notebook
    """

    pid = getpid()
    py = Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return f"RAM usage = {memory_use :.4} GB"
