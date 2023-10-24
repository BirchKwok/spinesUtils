"""工具文件"""
import os
import time

import numpy as np


def iter_count(file_name):
    """统计文件行数"""
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def drop_duplicates_with_order(list_):
    """删除列表中的重复项，只保留第一项"""
    assert isinstance(list_, (list, tuple, np.ndarray))
    indices = set()
    for idx in range(len(list_)-1, 0, -1):
        current_val = list_[idx]
        if current_val in list_[:idx]:
            indices.add(idx)
    return [list_[i] for i in range(len(list_)) if i not in indices]


def get_file_md5(filename):
    """获取文件的MD5值"""
    import hashlib
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def check_files_fingerprint(filename1, filename2):
    """比较两个文件指纹是否相等"""
    if get_file_md5(filename1) == get_file_md5(filename2):
        return True
    else:
        return False


def folder_iter(folder_path):
    """迭代取出父文件夹下所有子文件夹的文件"""
    assert os.path.isdir(folder_path)
    folder_path = 'model_tools'

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            yield file_path


def find_same_file(filename, to_find_path, absolute_path=False):
    """寻找对应目录下是否有相同指纹的文件"""

    flist = []

    for filename2 in folder_iter(to_find_path):
        if check_files_fingerprint(filename, filename2):
            if not absolute_path:
                flist.append(filename2)
            else:
                flist.append(os.path.abspath(filename2))
    
    if len(flist) > 0:
        return flist
    
    return


def is_in_ipython():
    """当前环境是否为ipython"""
    import sys

    is_ipython = 'ipykernel' in sys.modules or 'IPython' in sys.modules

    if is_ipython:
        return True
    else:
        return False


def reindex_iterable_object(obj, key=None, index_start=0):
    """对可迭代对象进行内部分组和重新索引

    :parameter
        obj: 可迭代对象
        key: 分组依据
        index_start: 索引起始值

    :return
        list[*tuple(index, value)]

    """
    from itertools import groupby

    sorted_obj = sorted(obj, key=key)  # 对列表进行排序
    grouped_obj = [list(group) for key, group in groupby(sorted_obj, key=key)]  # 对排序后的列表进行分组

    for group in grouped_obj:
        yield [(idx, g) for idx, g in enumerate(group, start=index_start)]


class Timer:
    """秒数为单位"""
    def __init__(self):
        self.start_time = None
        self.middle_points = []

    def check_is_start(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not started.")

    def start(self):
        """全局开始"""
        self.start_time = time.time()
        return self

    def middle_point(self):
        """中间点"""
        self.check_is_start()
        self.middle_points.append(time.time())
        return self

    def end(self):
        """全局结束"""
        self.check_is_start()

        time_diff = time.time() - self.start_time

        self.start_time = None
        self.middle_points = []
        return time_diff

    def last_timestamp_diff(self):
        """上一点时间"""
        self.check_is_start()

        if len(self.middle_points) == 0:
            return time.time() - self.start_time

        time_diff = time.time() - self.middle_points[-1]
        return time_diff

    def clear(self):
        """重置Timer"""
        self.check_is_start()

        self.start_time = None
        self.middle_points = []
        return self

    def sleep(self, secs):
        """睡眠"""
        self.check_is_start()

        time.sleep(secs)
        return self
