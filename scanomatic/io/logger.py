import threading
import time
import logging


class Logger(logging.Logger):
    _TO_FILE = None

    def __init__(self, name):
        super(Logger, self).__init__(name)
        self._custom_formatter = logging.Formatter(
            fmt="%(asctime)s -- %(levelname)s\t**%(name)s** %(message)s\n",
            datefmt='%Y-%m-%d %H:%M:%S',
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


def parse_log_file(path, seek=0, max_records=-1, filter_status=None):

    with open(path, 'r') as fh:

        if seek:
            fh.seek(seek)

        n = 0
        pattern = Logger.LOG_PARSING_EXPRESSION

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

            match = pattern.match(line)

            if match:

                if record and (filter_status is None or record['status'] in filter_status):
                    records.append(record)
                    n += 1
                groups = match.groups()
                record = {
                    'date': groups[0],
                    'time': groups[1],
                    'status': groups[2],
                    'source': groups[3],
                    'message': groups[4].strip()
                }
            elif record:
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
