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
        """
        A daemon for communications with main process.

        :param logger: A logging object
        :param parent_process: An object run in the main thread
        :param stdin: Path to stdin file
        :param stdout: Path to stdout file
        :param stderr: Path to stderr file

        Intended usage
        ==============

            import threading

            d = Communicator(...)
            t = threading.Thread(target=d.run)
            t.start()

        Where "..." signifies valid initiation parameters for the communicator.

        Parent Process
        ==============

        The parent process needs to have the following interface

        set_terminate()
        ---------------

        Method that will nicely break the main threads iteration

        set_pause()
        -----------

        Method that will pause main process' iteration
        Should return boolean success-value

        set_unpause()
        -------------

        Method that will resume main process' iteration
        Should return boolean success-value

        get_paused()
        ------------

        Should return boolean on if main thread is paused

        get_info()
        ----------

        Should return a string or collection of strings
        on general information about the process such as how
        it was initiated.

        get_current_step()
        ------------------

        Return integer for what iteration step main thread is on

        get_total_iterations()
        ----------------------

        Return the total number of iterations for the main thread

        get_progress()
        --------------

        Return float for progress, can be more detailed than simply
        taking the current step devided by total if such info is
        available.

        """
        self._parent = parent_process
        self._logger = logger
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr

        self._orphan = False
        self._running = False
        self._printing = False
        self._in_pos = None

        self._protocol = SUBPROC_COMMUNICATIONS()

        self.gated_print("DEAMON listens to", stdin)

    def gated_print(self, *args):
        """Safe printing that will not make both threads print at the
        same time"""

        while self._printing is True:
            time.sleep(0.02)

        self._printing = True
        for arg in args:
            print arg,
        print
        self._printing = False

    def set_terminate(self):

        self._running = False

    def run(self):
        """The main process for the communications daemon"""

        self._running = True
        self.gated_print("DEAMON is running")

        while self._running:

            lines = self._get_lines()

            for line in lines:

                output = self._parse(line.strip())

                if output is not None:

                    if not(isinstance(output, types.StringTypes)):
                        output = self._protocol.join(output)

                    output += (self._protocol.NEWLINE +
                               self._protocol.COMMUNICATION_END)

                    self.gated_print(output)

                    if not self._orphan:

                        try:
                            fs = open(self._stdout, 'r')
                            lines = fs.read().split()
                            fs.close()
                        except:
                            lines = ""

                        if output not in lines:

                            try:
                                stdout = open(self._stdout, 'a', 0)
                                sys.stdout = _Unbuffered_IO(stdout)
                                stderr = open(self._stderr, 'a', 0)
                                sys.stderr = _Unbuffered_IO(stderr)
                                self._orphan = True
                            except:
                                pass

                            self.gated_print(output)

            time.sleep(0.42)

    def _parse(self, line):
        """Looks up what was communicated to the subprocesses.

        Either responds directly or requests information from
        the main thread"""

        #
        # CONTROLS
        #

        #TERMINATING
        if line == self._protocol.TERMINATE:
            output = self._protocol.TERMINATING
            self._running = False
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
