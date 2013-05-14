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
import os
import re
from subprocess import Popen, PIPE
from itertools import chain

#
# INTERNAL DEPENDENCIES
#

import subproc_interface
from src.subprocs.protocol import SUBPROC_COMMUNICATIONS

#
# EXCEPTIONS
#


class BadCommunicateReturn(Exception):
    pass


class InvalidProcesCreationCall(Exception):
    pass


class AttemptedProcessOverride(Exception):
    pass


#
# METHODS
#


def _get_pinnings_str(pinning_list):

    pinning_string = ""

    for p in pinning_list:

        if p is None:

            pinning_string += "None,"

        else:

            try:
                pinning_string += "{0}x{1},".format(*p)
            except:
                pinning_string += "None,"

    return pinning_string[:-1]

#
# CLASSES
#


class _Proc_File_IO(object):

    def __init__(self, stdin, stdout, stderr, logger=None):
        """Proc File IO is a fake process.

        It exposes the neccesary interface common with true
        subprocesses. However it communicates with said subprocess
        not through PIPEs but via reading and writing to files.
        """
        self._PROC_COMM = SUBPROC_COMMUNICATIONS()
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

        t_string = self._PROC_COMM.PING
        t_string += self._PROC_COMM.VALUE_EXTEND.format(time.time())
        t_string += self._PROC_COMM.NEWLINE

        lines = self._get_feedback(t_string)

        if t_string in lines:
            retval = None

        return retval

    def communicate(self, c):
        """communicate emulates subprocess.communicate()"""
        return self._get_feedback(c)

    def _get_output_since_last_time(self, p, feed):
        #FIXIT is any in here needed?
        lines = ""
        try:
            fh = open(p[feed], 'r')
            if 'output-pos' in p:
                fh.seek(p['output-pos'])
            lines = fh.read()
            p['output-pos'] = fh.tell()
            fh.close()
        except:
            pass

        return lines

    def _get_feedback(self, c):

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

    def __init__(self, proc_type, top_controler, proc=None):
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
        self._PROC_COMM = SUBPROC_COMMUNICATIONS()
        self._proc_type = proc_type
        self._tc = top_controler
        self._proc = proc
        self._launch_param = None
        self._total = None
        self._exit_code = None
        self._stdin = None
        self._stdout = None
        self._stderr = None

    def get_type(self):
        """Returns the process type"""

        return self._proc_type

    def is_done(self):
        """Returns if process is done"""

        self._exit_code = self._proc.poll()
        return self._exit_code is not None

    def is_paused(self):
        """Returns is process is paused"""

        return (self.communicate(self._pre_comm(self._PROC_COMM.IS_PAUSED)) ==
                self._PROC_COMM.PAUSED_RESPONSE)

    def get_parameters(self):
        """Returns the parameters used to invoke the process"""

        if self._launch_param is None:
            self._launch_param = self._parse_parameters(
                self._proc.communicate(self._pre_comm(
                self._PROC_COMM.INFO)))

        return self._launch_param

    def get_prefix(self):
        """Returns the prefix that the project was created with"""

        if self._launch_param is None:
            self.get_parameters()

        if 'prefix' in self._launch_param:
            return self._launch_param['prefix']
        else:
            return ""

    def get_exit_code(self):

        if self._exit_code is not None:
            return self._exit_code
        else:
            return 0

    def get_progress(self):
        """Returns the progress (as precent).

        Note that this can be different than doing
        proc.get_current()/proc.get_total()
        since the subprocess is free to report a more detailed
        progress than just what iteration step it is on.
        """

        val = self._proc.communicate(self._pre_comm(
            self._PROC_COMM.PROGRESS))
        return self._get_val(val, self._PROC_COMM.PROGRESS, float)

    def get_current(self):
        """Returns the current iteration step number"""
        val = self._proc.communicate(self._pre_comm(self._PROC_COMM.CURRENT))
        return self._get_val(val, self._PROC_COMM.CURRENT, int)

    def get_total(self):
        """Returns the total iteration steps"""

        if self._total is None:
            val = self._proc.communicate(self._pre_comm(self._PROC_COMM.TOTAL))
            self._total = self._get_val(val, self._PROC_COMM.TOTAL, int)

        return self._total

    def spawn_proc(self, param_list):

        stdin, stdout, stderr = self.set_log_paths()

        if stdin is None:
            stdin = PIPE

        proc = Popen(map(str, param_list), stdin=stdin, stdout=stdout,
                     stderr=stderr, shell=False)
        return proc

    def set_process(self, proc):
        """Sets the process, only allowed if not yet set"""

        if self._proc is None:
            self._proc = proc
        else:
            raise AttemptedProcessOverride(self)

    def set_pause(self):
        """Requests that the subprocess pauses its operations"""

        return (self._proc.communicate(self._pre_comm(self._PROC_COMM.PAUSE))
                == self._PROC_COMM.PAUSING)

    def set_terminate(self):
        """Requests that the subprocess terminates"""

        return (self._proc.communicate(self._pre_comm(
            self._PROC_COMM.TERMINATE)) == self._PROC_COMM.TERMINATING)

    def set_unpause(self):
        """Requests that the subprocess resumes its operations"""

        return (self._proc.communicate(self._pre_comm(
            self._PROC_COMM.UNPAUSE)) == self._PROC_COMM.RUNNING)

    def _pre_comm(self, msg):
        return msg + self._PROC_COMM.NEWLINE

    def close_communications(self):

        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None

        if self._stdout is not None:
            self._stdout.close()
            self._stdout = None

        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def _get_val(self, ret_string, expected_start, dtype):
        """Help method for evaluating and validating the response
        from the subprocess"""

        len_exp_start = len(expected_start)

        if ((len_exp_start < len(ret_string)) and
                (ret_string[:len_exp_start] == expected_start)):

            return dtype(ret_string[len_exp_start + 1:])

        raise BadCommunicateReturn(ret_string, expected_start)
        return None

    def _parse_parameters(self, psm_in_text):

        #FIXIT  rewrite to fit new paradigm
        psm = model_experiment.copy_model(
            model_experiment.specific_project_model)

        psm_prefix = re.findall(r'__PREFIX__ (.*)', psm_in_text)
        if len(psm_prefix) > 0:
            psm['experiment-prefix'] = psm_prefix[0]

        psm_fixture = re.findall(r'__FIXTURE__ (.*)', psm_in_text)
        if len(psm_fixture) > 0:
            psm['fixture'] = psm_fixture[0]

        psm_scanner = re.findall(r'__SCANNER__ (.*)', psm_in_text)
        if len(psm_scanner) > 0:
            psm['scanner'] = psm_scanner[0]

        psm_root = re.findall(r'__ROOT__ (.*)', psm_in_text)
        if len(psm_root) > 0:
            psm['experiments-root'] = psm_root[0]

        psm_pinning = re.findall(r'__PINNING__ (.*)', psm_in_text)
        if len(psm_pinning) > 0:
            psm['pinnings-list'] = map(tuple, eval(psm_pinning[0]))

        psm_interval = re.findall(r'__INTERVAL__ (.*)', psm_in_text)
        if len(psm_interval) > 0:
            psm['interval'] = float(psm_interval[0])

        psm_scans = re.findall(r'__SCANS__ ([0-9]*)', psm_in_text)
        if len(psm_scans) > 0:
            psm['scans'] = int(psm_scans[0])

        psm_init_time = re.findall(r'__INIT-TIME__ (.*)', psm_in_text)
        if len(psm_init_time) > 0:
            start_time = float(psm_init_time[0])
        else:
            start_time = None

        psm_cur_image = re.findall(r'__CUR-IM__ ([0-9])', psm_in_text)
        if len(psm_cur_image) > 0:
            current_progress = int(psm_cur_image[0])
        else:
            current_progress = None

        if psm['interval'] is not None and psm['scans'] is not None:
            psm['duration'] = psm['interval'] * psm['scans'] / 60.0

        if 'experiment-prefix' in psm:
            self._logger.info('Revived experiment {0}'.format(
                psm['experiment-prefix']))


class Experiment_Scanning(_Subprocess):

    def __init__(self, top_controller, **params):

        super(Experiment_Scanning, self).__init__(
            subproc_interface.EXPERIMENT_SCANNING,
            top_controller)

        self._scanner = None
        self._new_proc = False

        self.set_process(self.get_proc(**params))

        #Give GUIs Scanner a new UUID so it cant talk to it
        if self._scanner is not None and self._new_proc:
            self._scanner.set_uuid()

    def get_param_list(self, sm):

        tc = self._tc
        e_list = [tc.paths.experiment]
        scanner = tc.scanners[sm['scanner']]
        self._scanner = scanner

        experiment_query = tc.config.get_default_experiment_query()
        experiment_query['-f'] = sm['fixture']
        experiment_query['-s'] = sm['scanner']
        experiment_query['-i'] = sm['interval']
        experiment_query['-n'] = sm['scans']

        if sm['experiments-root'] != '':
            experiment_query['-r'] = sm['experiments-root']

        experiment_query['-p'] = sm['experiment-prefix']
        experiment_query['-d'] = sm['experiment-desc']
        experiment_query['-c'] = sm['experiment-id']
        experiment_query['-l'] = sm['experiment-scan-layout-id']
        experiment_query['-u'] = scanner.get_uuid()

        experiment_query['-m'] = _get_pinnings_str(sm['pinnings-list'])

        self._launch_param = experiment_query

        #Make list of key & value pairs
        e_list += list(chain.from_iterable(experiment_query.items()))

        return e_list

    def get_proc(self, is_running=False, sm=None, **params):

        if is_running:

            raise InvalidProcesCreationCall("Not able to reconnect yet")

        else:

            if sm is not None:

                self._new_proc = True
                proc = self.spawn_proc(self.get_param_list(sm))

            else:

                raise InvalidProcesCreationCall(
                    "Cannot create experiment with specific model 'sm'")

        return proc

    def set_log_paths(self):

        tc = self._tc
        scanner = self._scanner

        stdin_path = tc.paths.experiment_stdin.format(
            tc.paths.get_scanner_path_name(scanner.get_name()))

        stdin = open(stdin_path, 'w')
        if self._new_proc:
            stdin.close()
            stdin = None

        stdout_path = tc.paths.log_scanner_out.format(scanner.get_socket())
        stdout = open(stdout_path, 'w')

        stderr_path = tc.paths.log_scanner_err.format(scanner.get_socket())
        stderr = open(stderr_path, 'w')

        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr

        return stdin, stdout, stderr


class Experiment_Rebuild(_Subprocess):

    def __init__(self, top_controller, **params):

        super(Experiment_Rebuild, self).__init__(
            subproc_interface.EXPERIMENT_REBUILD,
            top_controller)
        #FIXIT params should determine if subprocess.Popen be run
        #or fake process be created


class Analysis(_Subprocess):

    def __init__(self, top_controller, **params):
        """Analysis Subprocess wrapper.

        Spawn New Proc
        ==============

        For the class to spawn a new process it needs be
        initiated with either an a_dict or the experiment_root
        and experiment_prefix:

        a_dict
        ------

        A dictionary holding flag and value pairs for how to run
        the analysis.

        experiment_root
        ---------------

        Path to the root in which the experiement catalogue resides

        experiment_prefix
        -----------------

        The prefix used for the experiment.

        Reconnect To Old Proc
        =====================

        To reconnect to allready running process, class needs
        to be initiated with the following parameters:

        is_running
        ----------

        True


        """

        super(Analysis, self).__init__(
            subproc_interface.ANALYSIS,
            top_controller)

        self.set_process(self.get_proc(**params))

    def get_param_list(self, experiments_root, experiment_prefix, a_dict=None):

        if a_dict is None:
            a_dict = self._tc.config.get_default_analysis_query()

        a_list = [self._tc.paths.analysis]

        if '-i' not in a_dict or a_dict['-i'] is None:

            proc_name = os.path.join(
                experiments_root,
                experiment_prefix,
                self._tc.paths.experiment_first_pass_analysis_relative.format(
                    experiment_prefix))

            a_dict['-i'] = proc_name

        a_list += list(chain.from_iterable(a_dict.items()))

        return a_list

    def get_proc(self, is_running=False, a_dict=None,
                 experiments_root=None, experiment_prefix=None):

        if is_running:
            #Reconnect to running analyssi
            raise InvalidProcesCreationCall("Not able to reconnect yet")

        else:
            #Spawn new process
            if ((experiments_root is not None and
                 experiment_prefix is not None) or (a_dict is not None)):

                param_list = self.get_param_list(experiments_root,
                                                 experiment_prefix,
                                                 a_dict=a_dict)

                proc = self.spawn_proc(param_list)

            else:

                raise InvalidProcesCreationCall(
                    "Missing values for experiments_root or experiment_prefix")

        return proc

    def set_log_paths(self):
        """Sets a new out/err log file pair"""

        stdin = None
        stdout_path, stderr_path = self._tc.paths.get_new_log_analysis()
        stdout = open(stdout_path, 'w')
        stderr = open(stderr_path, 'w')

        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr

        return stdin, stdout, stderr
