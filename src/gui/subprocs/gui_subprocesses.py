#!/usr/bin/env python
"""The Subprocess-objects used by the mail program to communicate
with the true subprocesses"""
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

#
# INTERNAL DEPENDENCIES
#

import subproc_interface

#
# EXCEPTIONS
#


class BadCommunicateReturn(Exception):
    pass

#
# CLASS
#


class _SUBPROC_COMMUNICATIONS(object):

    IS_PAUSED = "__IS_PAUSED__"
    PAUSE = "__PAUSE__"
    PAUSING = "__PAUSING__"
    LAUNCH_PARAM = "__PARAM__"
    CURRENT = "__CURRENT__"
    TOTAL = "__TOTAL__"
    PROGRESS = "__PROGRESS__"
    TERMINATE = "__TERMINATE__"
    TERMINATING = "__TERMINATING__"
    UNPAUSE = "__UNPAUSE__"
    RUNNING = "__RUNNING__"
    PING = "__ECHO__ {0}"
    COMMUNICATION_END = "__DONE__"


class _Proc_File_IO(object):

    def __init__(self, stdin, stdout, stderr, logger=None):
        """Proc File IO is a fake process.

        It exposes the neccesary interface common with true
        subprocesses. However it communicates with said subprocess
        not through PIPEs but via reading and writing to files.
        """
        self._PROC_COMM = _SUBPROC_COMMUNICATIONS()
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._logger = logger

    def poll(self):
        """poll emulates a subprocess.poll()
        Returns None while still running,
        If stopped it returns Non-None
        """
        retval = 0

        t_string = self._PROC_COMM.PING.format(time.time())
        lines = self._get_feedback(t_string)

        if t_string in lines:
            retval = None

        return retval

    def communicate(self, c):
        """communicate emulates subprocess.communicate()"""
        return self._get_feedback(c)

    def _get_feedback(self, c):

        if c[-1] != "\n":
            c += "\n"

        try:
            fh = open(self.stdout, 'r')
            fh.read()
            fh_pos = fh.tell()
            fh.close()
        except:
            fh_pos = None

        try:
            fh = open(self.stdin, 'a')
            fh.write(c)
            fh.close()
        except:
            self._logger.error('Could not write to stdin')

        lines = ""
        i = 0

        #self._logger.info('stdout pos: {0}, sent to stdin: {1}'.format(
        #    fh_pos, c))

        while i < 10 and self._PROC_COMM.COMMUNICATION_END not in lines:

            try:
                fh = open(self.stdout, 'r')
                if fh_pos is not None:
                    fh.seek(fh_pos)
                lines += fh.read()
                fh_pos = fh.tell()
                fh.close()
            except:
                self._logger.error('Could not read stdout')

            if self._PROC_COMM.COMMUNICATION_END not in lines:
                time.sleep(0.1)
                i += 1

        #self._logger.info('stdout pos: {0}, got response: "{1}"'.format(
        #    fh_pos, lines))

        return lines


class _Subprocess(subproc_interface.SubProc_Interface):

    def __init__(self, proc_type, proc):
        """_Subprocess is a common implementation for the different
        subprocess objects that the gui uses to check status and
        communicate with said processes.

        It implements all SubProc_Interface methods and is thus
        ready to use without the need to further extending it.

        However, it is not intended to be invoked directly,
        but rather used as a superclass itself.

        init
        ====

        :param proc_type: One of the valid process types.
        :param proc: A true process object or a fake one


        proc
        ----

        The proc needs to implement the following methods.

        poll
        ....

        Poll should return None while still running else
        return a not none indicating the way it exited

        communicate
        ...........

        Communicate should accept a string as a parameter
        and should return a string response.
        """
        self._PROC_COMM = _SUBPROC_COMMUNICATIONS()
        self._proc_type = proc_type
        self._proc = proc
        self._launch_param = None
        self._total = None

    def get_type(self):
        """Returns the process type"""

        return self._proc_type

    def is_done(self):
        """Returns if process is done"""

        return self._proc.poll() is None

    def is_paused(self):
        """Returns is process is paused"""

        return (self.communicate(self.PROC_COMM.IS_PAUSED) ==
                self.PROC_COMM.PAUSED_RESPONSE)

    def get_parameters(self):
        """Returns the parameters used to invoke the process"""

        if self._launch_param is None:
            self._launch_param = self._proc.communicate(
                self.PROC_COMM.LAUNCH_PARAM)

        return self._launch_param

    def get_progress(self):
        """Returns the progress (as precent).

        Note that this can be different than doing
        proc.get_current()/proc.get_total()
        since the subprocess is free to report a more detailed
        progress than just what iteration step it is on.
        """

        val = self._proc.communicate(self.PROC_COMM.PROGRESS)
        return self._get_val(val, self.PROC_COMM.PROGRESS, float)

    def get_current(self):
        """Returns the current iteration step number"""
        val = self._proc.communicate(self.PROC_COMM.CURRENT)
        return self._get_val(val, self.PROC_COMM.CURRENT, int)

    def get_total(self):
        """Returns the total iteration steps"""

        if self._total is None:
            val = self._proc.communicate(self.PROC_COMM.TOTAL)
            self._total = self._get_val(val, self.PROC_COMM.TOTAL, int)

        return self._total

    def set_pause(self):
        """Requests that the subprocess pauses its operations"""

        return (self._proc.communicate(self.PROC_COMM.PAUSE) ==
                self.PROC_COMM.PAUSING)

    def set_terminate(self):
        """Requests that the subprocess terminates"""

        return (self._proc.communicate(self.PROC_COMM.TERMINATE) ==
                self.PROC_COMM.TERMINATING)

    def set_unpause(self):
        """Requests that the subprocess resumes its operations"""

        return (self._proc.communicate(self.PROC_COMM.UNPAUSE) ==
                self.PROC_COMM.RUNNING)

    def _get_val(self, ret_string, expected_start, dtype):
        """Help method for evaluating and validating the response
        from the subprocess"""

        len_exp_start = len(expected_start)

        if ((len_exp_start < len(ret_string)) and
                (ret_string[:len_exp_start] == expected_start)):

            return dtype(ret_string[len_exp_start + 1:])

        raise BadCommunicateReturn(ret_string, expected_start)
        return None


class Experiment_Scanning(_Subprocess):

    def __init__(self, **params):

        #FIXIT params should determine if subprocess.Popen be run
        #or fake process be created
        super(Experiment_Scanning, self).__init__(
            subproc_interface.EXPERIMENT_SCANNING,
            proc)


class Experiment_Rebuild(_Subprocess):

    def __init__(self, **params):

        #FIXIT params should determine if subprocess.Popen be run
        #or fake process be created
        super(Experiment_Rebuild, self).__init__(
            subproc_interface.EXPERIMENT_REBUILD,
            proc)


class Analysis(_Subprocess):

    def __init__(self, **params):

        #FIXIT params should determine if subprocess.Popen be run
        #or fake process be created
        super(Analysis, self).__init__(
            subproc_interface.ANALYSIS,
            proc)
