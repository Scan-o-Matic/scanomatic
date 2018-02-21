import threading
import time
import logging


class Logger(logging.Logger):

    def __init__(self, name):
        super(Logger, self).__init__(name)
        f = logging.Formatter(
            fmt="%(asctime)s -- %(levelname)s\t**%(name)s** %(message)s\n",
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        h = logging.StreamHandler()
        h.formatter = f
        self.addHandler(h)
        self.setLevel(logging.INFO)


class _ExtendedFileObject(file):
    # TODO: Regain buffer and release threads while closing

    def __init__(self, path, mode, buffering=None):

        super(_ExtendedFileObject, self).__init__(path, mode, buffering=buffering)
        self._semaphor = False
        self._buffer = []
        self._flush_buffer = False
        self._cache = []

    def pause(self):

        if -1 not in self._buffer:
            self._buffer.insert(0, -1)

        while self._semaphor:
            time.sleep(0.01)

    def close(self):

        if self._buffer and not self._buffer[0] == -1:
            self.pause()
        super(_ExtendedFileObject, self).close()
        self._flush_buffer = True
        self.resume()
        while self._buffer:
            time.sleep(0.01)
        return self._cache

    def resume(self):

        if self._buffer and self._buffer[0] == -1:
            self._buffer.pop(0)

    def write(self, s):
        self._write((s,))

    def writelines(self, *lines):

        if len(lines) == 1 and (isinstance(lines[0], list) or isinstance(lines[0], tuple)):
            lines = lines[0]

        self._write(lines)

    def _write(self, obj):

        id = hash(obj)
        self._buffer.append(id)
        t = threading.Thread(target=self._write_to_file, args=(id, obj))
        t.start()

    def _write_to_file(self, id, obj):

        while self._semaphor and id != self._buffer[0]:
            time.sleep(0.01)

        if self._flush_buffer:
            for l in obj:
                self._cache.append(obj)
            self._buffer.remove(id)
            return

        self._semaphor = True
        self._buffer.remove(id)

        super(_ExtendedFileObject, self).writelines([l if l.endswith(u"\n") else l + u"\n" for l in obj])
        self._semaphor = False


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
