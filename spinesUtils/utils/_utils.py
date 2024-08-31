import hashlib
import os


def iter_count(file_name):
    """
    Count the number of lines in a file efficiently by reading in large blocks.

    Parameters
    ----------
    file_name : str
        The path to the file whose lines are to be counted.

    Returns
    -------
    int
        The number of lines in the file.

    Notes
    -----
    This function reads the file in blocks (default size: 1 MB) to efficiently
    handle large files.

    Examples
    --------
    >>> iter_count('large_file.txt')
    1000000
    """
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def drop_duplicates_with_order(seq):
    """Remove duplicate elements from a sequence while preserving the order of the remaining elements.

    Parameters
    ----------
    seq : list, tuple, or ndarray
        The sequence from which duplicate elements are to be removed.

    Returns
    -------
    list
        A list containing the elements of the original sequence with duplicates removed.

    Notes
    -----
    This function iterates over the input sequence in reverse order, allowing it to
    remove duplicates while preserving the order of the first occurrences.

    Examples
    --------
    >>> drop_duplicates_with_order([1, 2, 2, 3, 2, 4, 3])
    [1, 2, 3, 4]
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_file_md5(filename):
    """
    Calculate and return the MD5 hash of a file.

    Parameters
    ----------
    filename : str
        The path to the file whose MD5 hash is to be computed.

    Returns
    -------
    str or None
        The MD5 hash of the file as a hexadecimal string. Returns None if an error occurs.

    Examples
    --------
    >>> get_file_md5("example.txt")
    'd41d8cd98f00b204e9800998ecf8427e'

    Note: The actual output will vary depending on the content of 'example.txt'.
    """
    try:
        md5 = hashlib.md5()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5.update(chunk)
        return md5.hexdigest()
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        return None


def check_files_fingerprint(filename1, filename2):
    """
    Check if two files have the same MD5 fingerprint (hash).

    Parameters
    ----------
    filename1 : str
        The path to the first file.
    filename2 : str
        The path to the second file.

    Returns
    -------
    bool
        True if both files have the same MD5 hash, False otherwise.

    Examples
    --------
    >>> check_files_fingerprint("file1.txt", "file2.txt")
    False

    Note: The actual output will depend on the contents of 'file1.txt' and 'file2.txt'.
    """
    return get_file_md5(filename1) == get_file_md5(filename2)


def folder_iter(folder_path):
    """
    Iterate over all files in a given folder, including files in subfolders.

    Parameters
    ----------
    folder_path : str
        The path to the folder.

    Yields
    ------
    str
        The path to each file within the folder.

    Raises
    ------
    ValueError
        If the folder_path is not a directory.

    Examples
    --------
    >>> for file_path in folder_iter("sample_folder"):
    ...     print(file_path)
    sample_folder/file1.txt
    sample_folder/subfolder/file2.txt
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a directory.")

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def find_same_file(filename, to_find_path, absolute_path=False):
    """
    Find files with the same MD5 fingerprint as the given file in a specified directory.

    Parameters
    ----------
    filename : str
        The path to the file to compare.
    to_find_path : str
        The path to the directory where to search for files.
    absolute_path : bool, default=False
        If True, returns absolute paths of the files found.

    Returns
    -------
    list of str
        A list of paths to files that have the same MD5 hash as the given file.

    Examples
    --------
    >>> find_same_file("example.txt", "to_find_folder")
    ['to_find_folder/example_copy.txt']

    Note: The actual output will depend on the contents of 'example.txt' and the files in 'to_find_folder'.
    """
    flist = []
    for filename2 in folder_iter(to_find_path):
        if check_files_fingerprint(filename, filename2):
            flist.append(os.path.abspath(filename2) if absolute_path else filename2)
    return flist


def is_in_ipython():
    """Identify whether the code is running in an IPython environment.

    Returns
    -------
    bool
        True if the code is running in an IPython environment, False otherwise.
    """

    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def reindex_iterable_object(obj, key=None, index_start=0):
    """Groups and reindexes elements within an iterable object.
    It sorts the elements based on a given key, groups them by this key, and enumerates each element in its group
        from a specified starting index.

    Parameters
    ----------
    obj : iterable object
        The object to be reindexed.
    key : function, optional
        A function that returns a key to group the elements by. If not specified, the elements will be grouped by
            themselves.
    index_start : int, optional
        The starting index for each element in its group.

    Returns
    -------
    generator
        A generator that yields the reindexed elements.

    Examples
    --------
    >>> # Example 1: Group by Odd and Even Numbers
    >>> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> reindexed = list(reindex_iterable_object(numbers, key=lambda x: x % 2 == 0))
    >>> for group in reindexed:
    ...     print(group)

    [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)]
    [(0, 2), (1, 4), (2, 6), (3, 8)]

    >>> # Example 2: Group by the First Letter of a String
    >>> strings = ['apple', 'banana', 'pear', 'grape', 'orange', 'watermelon']
    >>> reindexed = list(reindex_iterable_object(strings, key=lambda x: x[0]))
    >>> for group in reindexed:
    ...     print(group)

    [(0, 'apple')]
    [(0, 'banana')]
    [(0, 'grape')]
    [(0, 'orange')]
    [(0, 'pear')]
    [(0, 'watermelon')]

    >>> # Example 3: Group by the Length of a String
    >>> strings = ['apple', 'banana', 'peach', 'grape', 'orange', 'watermelon']
    >>> reindexed = list(reindex_iterable_object(strings, key=lambda x: len(x)))
    >>> for group in reindexed:
    ...     print(group)

    [(0, 'apple'), (1, 'peach'), (2, 'grape')]
    [(0, 'banana'), (1, 'orange')]
    [(0, 'watermelon')]

    """
    from itertools import groupby

    sorted_obj = sorted(obj, key=key)  # sort the object based on the key
    grouped_obj = [list(group) for key, group in groupby(sorted_obj, key=key)]  # group the object by the key

    for group in grouped_obj:
        yield [(idx, g) for idx, g in enumerate(group, start=index_start)]


def get_env_variable(name, default=None, default_type=str):
    """
    Retrieves an environment variable, casts its value to a specified type, and returns it.
    If the environment variable is not set or if the type casting fails, a default value is returned.

    Parameters
    ----------
    name : str
        The name of the environment variable to retrieve.
    default : optional
        The default value to return if the environment variable is not set or if type casting fails.
        Default is None.
    default_type : type, optional (default=str)
        The type to which the environment variable's value should be cast. Default is str.

    Returns
    -------
    default_type or type of 'default'
        The value of the environment variable cast to 'default_type', or the 'default' value
        if the environment variable is not set or casting fails.

    Examples
    --------
    # Assuming the environment variable 'EXAMPLE_VAR' is not set
    >>> get_env_variable('EXAMPLE_VAR', default=10, default_type=int)
    10

    # Assuming the environment variable 'EXAMPLE_VAR' is set to '5'
    >>> get_env_variable('EXAMPLE_VAR', default=10, default_type=int)
    5

    # Assuming the environment variable 'EXAMPLE_VAR' is set to 'invalid'
    >>> get_env_variable('EXAMPLE_VAR', default=10, default_type=int)
    10  # Returns the default value since casting fails
    """
    import os

    def type_cast(v):
        if v is None:
            return default  # if the value is None, return the default value
        try:
            # try to cast the value to the specified type
            return default_type(v)
        except Exception:
            return default

    value = os.environ.get(name)

    return type_cast(value)

