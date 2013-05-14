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
import types

#
# INTERNAL DEPENDENCIES
#

from src.subprocs.protocol import SUBPROC_COMMUNICATIONS

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

        self._protocol = SUBPROC_COMMUNICATIONS()

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

                    if not(isinstance(output, types.StringTypes)):
                        output = self._protocol.join(output)

                    output += (self._protocol.NEWLINE +
                               self._protocol.COMMUNICATION_END)

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

        #
        # CONTROLS
        #

        #TERMINATING
        if line == self._protocol.TERMINATE:
            output = self._protocol.TERMINATING
            self.__running = False
            self._parent.set_terminate()

        #PAUSE
        elif line == self._protocol.PAUSE:
            if self._parent.set_pause():
                output = self._protocol.PAUSING
            else:
                output = self._protocol.REFUSED

        #UNPAUSE
        elif line == self._protocol.UNPAUSE:
            if self._parent.set_unpause():
                output = self._protocol.RUNNING
            else:
                output = self._protocol.REFUSED

        #
        # PING
        #

        elif self._protocol.PING in line:

            output = line

        #
        # INFO REQUESTS
        #

        #ABOUT PROCESS
        elif line == self._protocol.INFO:
            output = self._parent.get_info()

        #CURRENT
        elif line == self._protocol.CURRENT:
            output = self._protocol.CURRENT
            output += self._protocol.VALUE_EXTEND.format(
                self._parent.get_current_step())

        #TOTAL
        elif line == self._protocol.TOTAL:
            output = self._protocol.TOTAL
            output += self._protocol.VALUE_EXTEND.format(
                self._parent.get_total_iterations())

        #PROGRESS
        elif line == self._protocol.PROGRESS:
            output = self._protocol.PROGRESS
            output += self._protocol.VALUE_EXTEND.format(
                self._parent.get_progress())

        #STATUS
        elif line == self._protocol.STATUS:
            if self._parent.get_paused():
                output = self._protocol.IS_PAUSED
            else:
                output = self._protocol.IS_RUNNING
        else:
            output = self._protocol.UNKNOWN
            output += self._protocol.VALUE_EXTEND.format(line)

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
