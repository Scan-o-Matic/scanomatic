#!/usr/bin/env python
"""Home-brewn logging tools"""
__author__ = "Martin Zackrisson, jetxee"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENICES
#

#import traceback
import datetime
import sys
import threading
import time
from functools import partial

#
# GLOBALS
#


#
# METHODS
#


def getLogger(name):

    return _Logger(name)


def setLogLevel(logLevelName):

    for k in _Logger._LOGLEVELS_TO_TEXT:
        if _Logger._LOGLEVELS_TO_TEXT[k] == logLevelName:
            _Logger._LOGLEVEL = k
            break


def setLogLevels(logLevels=None):

    if logLevels is not None:
        _Logger._LOGLEVELS = logLevels

    _Logger._LOGLEVELS_TO_TEXT = {}

    for lvl, lvlMethodName, lvlName in _Logger._LOGLEVELS:

        if lvl in _Logger._LOGLEVELS_TO_TEXT:
            raise Exception(
                "Duplicated logging level " +
                "{0}: {1} attempt to replace {2}".format(
                    lvl, _Logger._LOGLEVELS_TO_TEXT[lvl], lvlName))

        else:
            _Logger._LOGLEVELS_TO_TEXT[lvl] = lvlName


def setLoggingTarget(filePath, redirectStdOut=False, redirectStdErr=False,
                     writeMode='w'):

    class ExtendedFileObject(file):

        def __init__(self, filePath, writeMode):

            super(ExtendedFileObject, self).__init__(filePath, writeMode)
            self._semaphor = False
            self._buffer = []

        def write(self, s):

            self._buffer.append()
            self._write()

        def writelines(self, *lines):

            lines = list(lines)
            for i in range(len(lines)):
                if lines[i][-1] != "\n":
                    lines[i] = str(lines[-1]) + "\n"
            self._buffer += lines
            self._write()

        def _write(self):

            t = threading.Thread(target=self._writeToFile)
            t.start()

        def _writeToFile(self):

            while self._semaphor:
                time.sleep(0.01)

            self._semaphor = True
            curLength = len(self._buffer)
            super(ExtendedFileObject, self).writelines(*self._buffer[:curLength])
            for i in range(curLength):
                self._buffer.pop(0)
            self._semaphor = False

    global _LOGFILE

    _LOGFILE = ExtendedFileObject(filePath, writeMode)
    _Logger._LOGFILE = _LOGFILE

    if redirectStdOut:
        sys.stdout = _LOGFILE
    if redirectStdErr:
        sys.stderr = _LOGFILE


def getLevels():

    return [_Logger._LOGLEVELS_TO_TEXT[k] for k in
            sorted(_Logger._LOGLEVELS_TO_TEXT.keys())]


def print_more_exc():

    tb = sys.exc_info()[2]  # Get the traceback object

    while tb.tb_next:
        tb = tb.tb_next

    stack = []

    f = tb.tb_frame
    while f:
        stack.append(f)
        f = f.f_back

    stack.reverse()

    #traceback.print_exc()
    for frame in stack[:-3]:
        print "\n\tFrame {0} in {1} at line {2}".format(frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno)

    print "\n\n"


#
# CLASSES
#

class _Logger(object):

    _LOGLEVEL = 3
    _LOGFILE = None
    _LOGFORMAT = "%Y-%m-%d %H:%M:%S -- {name} -- {lvl}: "
    _LOGLEVELS = [
        (0, 'exception', 'EXCEPTION'),
        (1, 'critical', 'CRITICAL'),
        (2, 'error', 'ERROR'),
        (3, 'warning', 'WARNING'),
        (4, 'info', 'INFO'),
        (5, 'debug', 'DEBUG')]

    _LOGLEVELS_TO_TEXT = {}

    def __init__(self, loggerName):

        self._loggerName = loggerName

        if (len(self._LOGLEVELS_TO_TEXT) != len(self._LOGLEVELS)):
            setLogLevels()

        for lvl, lvlMethodName, lvlName in self._LOGLEVELS:

            if hasattr(self, lvlMethodName):
                raise Exception(
                    "Trying to mask existing method {0} not allowed".format(
                        lvlMethodName))

            else:

                setattr(self, lvlMethodName,
                        partial(_Logger._output, self, lvl))

    def _decorate(self, lvl):

        return datetime.datetime.now().strftime(self._LOGFORMAT).format(
            lvl=self._LOGLEVELS_TO_TEXT[lvl],
            name=self._loggerName)

    def _output(self, lvl, msg):

        if (lvl <= self._LOGLEVEL and self._LOGFILE is not None and lvl in
                self._LOGLEVELS_TO_TEXT):
            if isinstance(msg, list) or isinstance(msg, tuple):
                msg = list(msg)
                msg[0] = self._decorate(lvl) + msg[0]
            else:
                msg = self._decorate(lvl) + msg

            self._LOGFILE.writelines(msg)
