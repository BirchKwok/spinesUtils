"""工具文件"""
import os

import numpy as np


def iter_count(file_name):
    """统计文件行数"""
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def check_has_params(func, params):
    """检查函数是否有指定形参"""
    assert isinstance(params, str)
    import inspect
    sig = inspect.signature(func)
    param = sig.parameters.get(params, None)
    if param is not None:
        return True
    else:
        return False


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
