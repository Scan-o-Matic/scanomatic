from __future__ import absolute_import
import logging
import re


class Logger(logging.Logger):
    _TO_FILE = None

    def __init__(self, name):
        super(Logger, self).__init__(name)
        self._custom_formatter = logging.Formatter(
            fmt="%(asctime)s -- %(levelname)s **%(name)s** %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S %Z',
        )
        h = logging.StreamHandler()
        h.formatter = self._custom_formatter
        self.addHandler(h)
        self.setLevel(logging.INFO)
        if self._TO_FILE:
            self.log_to_file(**self._TO_FILE)

    def log_to_file(self, path, level=None):
        if level is None:
            level = self.level
        Logger._TO_FILE = {'path': path, 'level': level}

        fh = logging.FileHandler(path)
        fh.setLevel(level)
        fh.formatter = self._custom_formatter
        self.addHandler(fh)


PARSE_PATTERN = re.compile(
    r'^(\d{4}-\d{1,2}-\d{1,2}) (\d{1,2}:\d{1,2}:\d{1,2} \w*) -- (\w+) \*{2}(.+)\*{2} (.*)$'
)


def parse_log_file(path, seek=0, max_records=-1, filter_status=None):

    with open(path, 'r') as fh:

        if seek:
            fh.seek(seek)

        n = 0

        records = []
        tell = fh.tell()
        garbage = []
        record = {}
        eof = False
        while n < max_records or max_records < 0:

            line = fh.readline()
            if tell == fh.tell():
                eof = True
                break
            else:
                tell = fh.tell()

            match = PARSE_PATTERN.match(line)

            if match:
                groups = match.groups()
                record = {
                    'date': groups[0],
                    'time': groups[1],
                    'status': groups[2],
                    'source': groups[3],
                    'message': groups[4].strip()
                }

                if filter_status is None or record['status'] in filter_status:
                    records.append(record)
                    n += 1
            elif line:
                if record:
                    record['message'] += '\n{0}'.format(line.rstrip())
                else:
                    garbage.append(line.rstrip())

        return {
            'file': path,
            'start_position': seek,
            'end_position': tell,
            'end_of_file': eof,
            'records': records,
            'garbage': garbage
        }
