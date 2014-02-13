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
# CLASSES
#

class Logger(object):

    EXCEPTION = 0
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4
    DEBUG = 5

    _LOGFORMAT = "%Y-%m-%d %H:%M:%S -- {name} -- {lvl}: "

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

    def __init__(self, loggerName):

        self._level = self.INFO
        self._logFile = None
        self._loggerName = loggerName
        self._logLevelToMethod = {}
        self._usePrivateOutput = False
        self._suppressPrints = False

        if (len(self._LOGLEVELS_TO_TEXT) != len(self._LOGLEVELS)):
            Logger.SetLogLevels()

        for lvl, lvlMethodName, lvlName in self._LOGLEVELS:

            if hasattr(self, lvlMethodName):
                raise Exception(
                    "Trying to mask existing method {0} not allowed".format(
                        lvlMethodName))

            else:

                setattr(self, lvlMethodName,
                        partial(Logger._output, self, lvl))

                self._logLevelToMethod[lvl] = getattr(self, lvlMethodName)

    @classmethod
    def SetLogLevels(cls, logLevels=None):

        if logLevels is not None:
            cls._LOGLEVELS = logLevels

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
    def SetDefaultOutputTarget(
            cls,  target, catchStdOut=False, catchStdErr=False,
            writeMode='w'):

        if (cls._DEFAULT_LOGFILE is not None):
            cls._DEFAULT_LOGFILE.close()

        if (target is not None):
            cls._DEFAULT_LOGFILE = _ExtendedFileObject(target, writeMode)
        else:
            if cls._DEFAULT_LOGFILE is not None:
                cls._DEFAULT_LOGFILE.close()
            cls._DEFAULT_LOGFILE = None

        cls._DEFAULT_CATCH_OUT = catchStdOut
        cls._DEFAULT_CATCH_ERR = catchStdErr

        cls.UseDefaultCatching()

    @classmethod
    def UseDefaultCatching(cls):
        if (cls._DEFAULT_CATCH_ERR and cls._DEFAULT_LOGFILE is not None):
            sys.stderr = cls._DEFAULT_LOGFILE
        else:
            sys.stderr = sys.__stderr__

        if (cls._DEFAULT_CATCH_OUT and cls._DEFAULT_LOGFILE is not None):
            sys.stdout = cls._DEFAULT_LOGFILE
        else:
            sys.stdout = sys.__stdout__

    @classmethod
    def GetLevels(cls):

        return [cls._LOGLEVELS_TO_TEXT[k] for k in
                sorted(cls._LOGLEVELS_TO_TEXT.keys())]

    @property
    def usePrivateOutput(self):
        return self._usePrivateOutput

    @usePrivateOutput.setter
    def usePrivateOutput(self, value):

        if (not value):
            self.catchStdOut = False
            self.catchStdErr = False

        self._usePrivateOutput = value

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):

        if (isinstance(value, int)):
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
    def catchStdOut(self):

        return (sys.stdout == self._logFile or
                not self._usePrivateOutput and
                sys.stdout == self._DEFAULT_LOGFILE)

    @catchStdOut.setter
    def catchStdOut(self, value):
        if value and self._logFile is not None:
            sys.stdout = self._logFile
        elif self._DEFAULT_CATCH_OUT and self._DEFAULT_LOGFILE is not None:
            sys.stdout = self._DEFAULT_LOGFILE
        else:
            sys.stdout = self.__stdout__

    @property
    def catchStdErr(self):

        return (sys.stderr == self._logFile or
                not self._usePrivateOutput and
                sys.stderr == self.__stderr__)

    @catchStdErr.setter
    def catchStdErr(self, value):
        if value and self._logFile is not None:
            sys.stderr = self._logFile
        elif (self._DEFAULT_CATCH_ERR and self._DEFAULT_LOGFILE is not None):
            sys.stderr = self._DEFAULT_LOGFILE
        else:
            sys.stderr = self.__stderr__

    @property
    def supressPrints(self):

        return self._suppressPrints

    @supressPrints.setter
    def supressPrints(self, value):

        self._suppressPrints = value

    def _decorate(self, lvl):

        return datetime.datetime.now().strftime(self._LOGFORMAT).format(
            lvl=self._LOGLEVELS_TO_TEXT[lvl],
            name=self._loggerName)

    def _output(self, lvl, msg):

        if (lvl <= self._level and lvl in self._LOGLEVELS_TO_TEXT):

            output = (self._usePrivateOutput and self._logFile or
                      self._DEFAULT_LOGFILE)

            if (output is not None):
                if isinstance(msg, list) or isinstance(msg, tuple):
                    msg = list(msg)
                    msg[0] = self._decorate(lvl) + msg[0]
                else:
                    msg = self._decorate(lvl) + msg

                self._logFile.writelines(msg)

            elif (not self._suppressPrints):

                print self._decorate(lvl) + str(msg)

    def setOutputTarget(
            self, target, catchStdOut=False, catchStdErr=False,
            writeMode='w'):

        if (self._logFile is not None):
            self._logFile.close()

        if (target is not None):
            self._logFile = _ExtendedFileObject(target, writeMode)
        else:
            if (self._logFile is not None):
                self._logFile.close()
            self._logFile = None

        self.catchStdOut = catchStdOut
        self.catchStdErr = catchStdErr

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

    def __init__(self, filePath, writeMode):

        super(_ExtendedFileObject, self).__init__(filePath, writeMode)
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
        super(_ExtendedFileObject, self).writelines(self._buffer[:curLength])
        for i in range(curLength):
            self._buffer.pop(0)
        self._semaphor = False
