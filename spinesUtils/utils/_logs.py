import os
import sys
from datetime import datetime

from spinesUtils.asserts import TypeAssert


class Printer:
    @TypeAssert({'name': (None, str), 'fp': (None, str),
                 'verbose': (bool, int), 'truncate_file': bool, 'with_time': bool}, func_name='Printer')
    def __init__(self, name=None, fp=None, verbose=True, truncate_file=True, with_time=True):
        self.name = name

        self.fp = fp
        self.verbose = verbose
        self.with_time = with_time

        if truncate_file:
            self._truncate_file()

    def _truncate_file(self):
        if self.fp is not None:
            if os.path.isfile(self.fp):
                os.truncate(self.fp, 0)

    def prefix_format(self):
        if self.with_time:
            time_str = datetime.now().strftime("%H:%M:%S %Y-%m-%d") + ' - '
        else:
            time_str = ''

        if self.name is not None:
            return time_str + self.name + ' - '
        return time_str

    @TypeAssert({'string': str, 'access_way': str, 'line_end': (None, str)})
    def insert2file(self, string, access_way='a', line_end='\n'):
        assert access_way in ('a', 'w')

        import os

        if not os.path.isfile(self.fp):
            self.print(f"file {self.fp} not exists, will be created.")

        if line_end is not None:
            string += line_end

        with open(self.fp, access_way) as f:
            f.write(self.prefix_format() + string)

    @TypeAssert({'string': str, 'line_end': (None, str)})
    def print(self, string, line_end='\n'):
        if not self.verbose:
            return
        if self.verbose < 50:
            writer = sys.stderr.write
        else:
            writer = sys.stdout.write

        if line_end is not None:
            writer(self.prefix_format() + string + line_end)
        else:
            writer(self.prefix_format() + string)

    @TypeAssert({'string': str, 'access_way': str, 'line_end': (None, str)})
    def insert_and_throwout(self, string, access_way='a', line_end='\n'):
        self.insert2file(string, access_way, line_end)
        self.print(string)
