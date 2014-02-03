#!/usr/bin/env python
"""Communications module for scan-o-matic subprocess"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import time
import sys
import threading
import weakref

#
# INTERNAL DEPENDENCIES
#

from src.subprocs.io import Proc_IO, Unbuffered_IO
import scanomatic.io.logger as logger

#
# EXCEPTIONS
#

#
# CLASSES
#


class Communicator(object):

    def __init__(self, parent_process, stdin, stdout, stderr):
        """
        A daemon for communications with main process.

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
        self._parent = weakref.ref(parent_process) if parent_process else None
        self._logger = logger.Logger("Communicator")
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr

        self._orphan = False
        self._running = False
        self._printing = False
        self._in_pos = None

        self._io = None
        self._set_io()

        self._io.send("DEAMON listens to {0}".format(stdin))
        self._io.send("DEAMON prints to {0}".format(stdout))
        self._io.send("DEAMON errors to {0}".format(stderr))

    def set_terminate(self):

        self._running = False

    def _set_io(self):

        self._io = Proc_IO(self._stdout, self._stdin,
                           recieve_pos=0,
                           send_file_state='a')

        self._io.send("Redirecting stdout to dev/null")

        self._io.send("Errors print to error file {0}".format(self._stderr))
        if self._stderr is not None:
            stderr = open(self._stderr, 'a', 0)
            sys.stderr = Unbuffered_IO(stderr)
            sys.stdout = sys.stderr  # open(os.devnull, 'w')

        self._orphan = True

    def run(self):
        """The main process for the communications daemon"""

        curThread = threading.currentThread()
        nonMeThreads = [t for t in threading.enumerate() if
                        t is not curThread]

        if len(nonMeThreads) != 1:
            raise Exception("Uncertain threading situation")

        mainThread = nonMeThreads[0]

        self._running = True
        self._io.send("DEAMON is running")

        while self._running and mainThread.isAlive():

            self._io.recieve(self._parse)

            #self._io.send("Communicator is alive @ {0}".format(time.time()))
            time.sleep(0.42)

    def _respond(self, output, timestamp):

        self._io.send(self._io.decorate(output, timestamp))

        """
        output = self._parse(line.strip())

        if output is not None:

            if not(isinstance(output, types.StringTypes)):
                output = self._io.NEWLINE.join(output)

            output += (self._io.NEWLINE +
                        self._io.COMMUNICATION_END)

        """

    def _parse(self, decorated_msg):
        """Looks up what was communicated to the subprocesses.

        Either responds directly or requests information from
        the main thread"""

        timestamp, line = self._io.undecorate(decorated_msg)

        line = line.strip()

        self._io.send("Communicator got '{0}' w/ stamp {1}".format(
            line, timestamp))

        #
        # CONTROLS
        #

        #TERMINATING
        if line == self._io.TERMINATE:
            output = self._io.TERMINATING
            self._running = False
            self._parent().set_terminate()

        #PAUSE
        elif line == self._io.PAUSE:
            if self._parent().set_pause():
                output = self._io.PAUSING
            else:
                output = self._io.REFUSED

        #UNPAUSE
        elif line == self._io.UNPAUSE:
            if self._parent().set_unpause():
                output = self._io.UNPAUSING
            else:
                output = self._io.REFUSED

        #
        # PING
        #

        elif self._io.PING in line:

            output = line

        #
        # INFO REQUESTS
        #

        #ABOUT PROCESS
        elif line == self._io.INFO:
            output = self._parent().get_info()
            if output is not None and output[0] != self._io.INFO:
                output = (self._io.INFO, ) + output

        #CURRENT
        elif line == self._io.CURRENT:
            output = self._io.CURRENT
            output += self._io.VALUE_EXTEND.format(
                self._parent().get_current_step())

        #TOTAL
        elif line == self._io.TOTAL:
            output = self._io.TOTAL
            output += self._io.VALUE_EXTEND.format(
                self._parent().get_total_iterations())

        #PROGRESS
        elif line == self._io.PROGRESS:
            output = self._io.PROGRESS
            output += self._io.VALUE_EXTEND.format(
                self._parent().get_progress())

        #STATUS
        elif line == self._io.STATUS:
            if self._parent().get_paused():
                output = self._io.IS_PAUSED
            else:
                output = self._io.IS_RUNNING
        else:
            output = self._io.UNKNOWN
            output += self._io.VALUE_EXTEND.format(line)

        if output is not None:
            self._respond(output, timestamp)
