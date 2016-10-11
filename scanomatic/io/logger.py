# import traceback
import datetime
import sys
import threading
import time
from functools import partial
import warnings
import re

LOG_RECYCLE_TIME = 60 * 60 * 24

#
# CLASSES
#


class Logger(object):

    EXCEPTION = 0
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4
    DEBUG = 5

    _LOGFORMAT = "%Y-%m-%d %H:%M:%S -- {lvl}\t**{name}** "
    LOG_PARSING_EXPRESSION = re.compile(
        r"(\d{4}-\d{1,2}-\d{1,2}) (\d{1,2}:\d{1,2}:\d{1,2}) -- (\w+)\t\*{2}([^\*]+)\*{2}(.*)")

    _DEFAULT_LOGFILE = None
    _DEFAULT_CATCH_OUT = False
    _DEFAULT_CATCH_ERR = False

    _LOGLEVELS = [
        (0, 'exception', 'EXCEPTION'),
        (1, 'critical', 'CRITICAL'),
        (2, 'error', 'ERROR'),
        (3, 'warning', 'WARNING'),
        (4, 'info', 'INFO'),
        (5, 'debug', 'DEBUG')]

    _LOGLEVELS_TO_TEXT = {}

    def __init__(self, name, active=True):

        self._level = self.INFO
        self._log_file = None
        self._loggerName = name
        self._logLevelToMethod = {}
        self._usePrivateOutput = False
        self._suppressPrints = False
        self._active = active

        if len(self._LOGLEVELS_TO_TEXT) != len(self._LOGLEVELS):
            Logger.set_global_log_levels()

        for lvl, lvlMethodName, lvlName in self._LOGLEVELS:

            if hasattr(self, lvlMethodName):
                pass
            else:

                setattr(self, lvlMethodName,
                        partial(Logger._output, self, lvl))

                self._logLevelToMethod[lvl] = getattr(self, lvlMethodName)

    def exception(self, msg):

        self._output(0, msg)

    def critical(self, msg):

        self._output(1, msg)

    def error(self, msg):

        self._output(2, msg)

    def warning(self, msg):

        self._output(3, msg)

    def info(self, msg):

        self._output(4, msg)

    def debug(self, msg):

        self._output(5, msg)

    def pause(self):

        file = self._active_log_file
        if file is None:
            self.error("Attempting to pause logging while not having any log file")
        else:
            file.pause()

    def resume(self):

        file = self._active_log_file
        if file is None:
            self.error("Attempting to resume logging while not having any log file")
        else:
            file.resume()

    def close_output(self):

        file = self._active_log_file
        if file is not None:
            file.close()
            if file is self._log_file:
                self._log_file = None
            else:
                self._DEFAULT_LOGFILE = None

    @classmethod
    def set_global_log_levels(cls, log_levels=None):

        if log_levels is not None:
            cls._LOGLEVELS = log_levels

        cls._LOGLEVELS_TO_TEXT = {}

        for lvl, lvlMethodName, lvlName in cls._LOGLEVELS:

            if lvl in cls._LOGLEVELS_TO_TEXT:
                raise Exception(
                    "Duplicated logging level " +
                    "{0}: {1} attempt to replace {2}".format(
                        lvl, cls._LOGLEVELS_TO_TEXT[lvl], lvlName))

            else:
                cls._LOGLEVELS_TO_TEXT[lvl] = lvlName
                setattr(cls, lvlName, lvl)

    @classmethod
    def set_global_output_target(
            cls,  target, catch_stdout=False, catch_stderr=False,
            mode='w', buffering=None):

        if cls._DEFAULT_LOGFILE is not None:
            cls._DEFAULT_LOGFILE.close()

        if target is not None:
            cls._DEFAULT_LOGFILE = _ExtendedFileObject(target, mode,
                                                       buffering=512 if buffering is None else buffering)
        else:
            if cls._DEFAULT_LOGFILE is not None:
                cls._DEFAULT_LOGFILE.close()
            cls._DEFAULT_LOGFILE = None

        cls._DEFAULT_CATCH_OUT = catch_stdout
        cls._DEFAULT_CATCH_ERR = catch_stderr

        cls.use_global_logging_settings()

    @classmethod
    def use_global_logging_settings(cls):
        if cls._DEFAULT_CATCH_ERR and cls._DEFAULT_LOGFILE is not None:
            sys.stderr = cls._DEFAULT_LOGFILE
        else:
            sys.stderr = sys.__stderr__

        if cls._DEFAULT_CATCH_OUT and cls._DEFAULT_LOGFILE is not None:
            sys.stdout = cls._DEFAULT_LOGFILE
        else:
            sys.stdout = sys.__stdout__

    @classmethod
    def get_logging_levels(cls):

        return [cls._LOGLEVELS_TO_TEXT[k] for k in
                sorted(cls._LOGLEVELS_TO_TEXT.keys())]

    @property
    def use_private_output(self):
        return self._usePrivateOutput

    @use_private_output.setter
    def use_private_output(self, value):

        if not value:
            self.catch_stdout = False
            self.catch_stderr = False

        self._usePrivateOutput = value

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):

        if isinstance(value, int):
            assert value in self._LOGLEVELS_TO_TEXT.keys(), "Level unknown"
            self._level = value
            return

        else:

            for k, v in self._LOGLEVELS_TO_TEXT.items():

                if v.lower() == value.lower():

                    self._level = k
                    return

        raise Exception("Level unknown")

    @property
    def catch_stdout(self):

        return (sys.stdout == self._log_file or
                not self._usePrivateOutput and
                sys.stdout == self._DEFAULT_LOGFILE)

    @catch_stdout.setter
    def catch_stdout(self, value):
        if not value:
            sys.stdout = sys.__stdout__
        elif self._log_file is not None:
            sys.stdout = self._log_file
        elif self._DEFAULT_CATCH_OUT and self._DEFAULT_LOGFILE is not None:
            sys.stdout = self._DEFAULT_LOGFILE
        else:
            warnings.warn("No log file to redirect output into")

    @property
    def catch_stderr(self):

        return (sys.stderr == self._log_file or
                not self._usePrivateOutput and
                sys.stderr == self._DEFAULT_LOGFILE)

    @catch_stderr.setter
    def catch_stderr(self, value):
        if not value:
            sys.stderr = sys.__stderr__
        if value and self._log_file is not None:
            sys.stderr = self._log_file
        elif self._DEFAULT_CATCH_ERR and self._DEFAULT_LOGFILE is not None:
            sys.stderr = self._DEFAULT_LOGFILE
        else:
            warnings.warn("No log file to redirect errors into")

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def surpress_prints(self):

        return self._suppressPrints

    @surpress_prints.setter
    def surpress_prints(self, value):

        self._suppressPrints = value

    def _decorate(self, lvl):

        return datetime.datetime.now().strftime(self._LOGFORMAT).format(
            lvl=self._LOGLEVELS_TO_TEXT[lvl],
            name=self._loggerName)

    def _output(self, lvl, msg):

        if (self._active and lvl <= self._level and
                lvl in self._LOGLEVELS_TO_TEXT):

            output = (self._usePrivateOutput and self._log_file or
                      self._DEFAULT_LOGFILE)

            if output is not None:
                if isinstance(msg, list) or isinstance(msg, tuple):
                    msg = list(msg)
                    msg[0] = self._decorate(lvl) + msg[0]
                else:
                    msg = self._decorate(lvl) + msg

                self._log_file.writelines(msg)

            elif not self._suppressPrints:

                print self._decorate(lvl) + str(msg)

    def set_output_target(
            self, target, catch_stdout=False, catch_stderr=False,
            mode='w', buffering=None):

        if self._log_file is not None:
            self._log_file.close()

        if target is not None:
            self._log_file = _ExtendedFileObject(target, mode, buffering=512 if buffering is None else buffering)
        else:
            if self._log_file is not None:
                self._log_file.close()
            self._log_file = None

        self.catch_stdout = catch_stdout
        self.catch_stderr = catch_stderr

    def traceback(self, lvl=None):

        tb = sys.exc_info()[2]  # Get the traceback object

        while tb.tb_next:
            tb = tb.tb_next

        stack = []

        f = tb.tb_frame
        while f:
            stack.append(f)
            f = f.f_back

        stack.reverse()

        if lvl is None:
            lvl = max(self._LOGLEVELS_TO_TEXT.keys())
        output = self._logLevelToMethod[lvl]

        txt = "Traceback:\n" + "\n".join(
            ["\n\tFrame {0} in {1} at line {2}".format(
                frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno)
             for frame in stack[:-3]])

        output(txt)


class _ExtendedFileObject(file):

    def __init__(self, path, mode, buffering=None):

        super(_ExtendedFileObject, self).__init__(path, mode, buffering=buffering)
        self._semaphor = False
        self._buffer = []

    def write(self, s):

        self._buffer.append(s)
        self._write()

    def writelines(self, *lines):

        lines = list(lines)
        for i in range(len(lines)):
            if lines[i][-1] != "\n":
                lines[i] = str(lines[-1]) + "\n"
        self._buffer += lines
        self._write()

    def _write(self):

        t = threading.Thread(target=self._write_to_file)
        t.start()

    def _write_to_file(self):

        while self._semaphor:
            time.sleep(0.01)

        self._semaphor = True
        length = len(self._buffer)
        super(_ExtendedFileObject, self).writelines(self._buffer[:length])
        for i in range(length):
            self._buffer.pop(0)
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