#!/usr/bin/env python
"""Communications module for scan-o-matic subprocess"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import time
import sys


#
# CLASSES
#


class _Unbuffered_IO:

    def __init__(self, stream):
        """This class provides and unbuffered IO-writer"""
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class Communicator(object):

    def __init__(self, logger, parent_process, stdin, stdout, stderr):

        self._parent = parent_process
        self._logger = logger
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr

        self._oraphan = False
        self._running = False
        self._printing = False
        self._in_pos = None

        self._gated_print("DEAMON listens to", stdin)

    def _gated_print(self, *args):

        while self._printing is True:
            time.sleep(0.02)

        self._printing = True
        for arg in args:
            print arg,
        print
        self._printing = False

    def run(self):

        self._running = True
        self._gated_print("DEAMON is running")

        while self._running:

            lines = self._get_lines()

            for line in lines:

                output = self._parse(line.strip())

                if output is not None:

                    output += "\n__DONE__"
                    self._gated_print(output)

                    if not self._orphan:
                        fs = open(self._stdout, 'r')
                        lines = fs.read().split()
                        fs.close()
                        if output not in lines:

                            try:
                                stdout = open(self._stdout, 'a', 0)
                                sys.stdout = _Unbuffered_IO(stdout)
                                stderr = open(self._stderr, 'a', 0)
                                sys.stderr = _Unbuffered_IO(stderr)
                                self._orphan = True
                            except:
                                pass

                            self._gated_print(output)

            time.sleep(0.42)

    def _parse(self, line):

        if line == "__QUIT__":
            output = "DEAMON got quit request"

            self.__running = False
            self._parent.set_terminate()

        elif '__ECHO__' in line:

            output = line

        elif line == '__INFO__':
            output = self._parent.get_info()

        else:
            output = "DEAMON got unkown request '{0}'".format(line)

        return output

    def _get_lines(self):

        lines = []
        try:
            fs = open(self._stdin, 'r')
            if self._in_pos is not None:
                fs.seek(self._in_pos)
            lines = fs.readlines()
            self._in_pos = fs.tell()
            fs.close()
        except:
            #self._orphan = True
            self._logger.warning("Lost contact with outside world!")

        return lines
