#!/usr/bin/env python
"""The GUIs Communication Classes for GUI to Subprocess talking."""
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
import itertools
import inspect
import logging

#
# INTERNAL DEPENDENCIES
#

from src.subprocs.io import Proc_IO

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
# CONSTANTS
#

EXPERIMENT_SCANNING = 11
EXPERIMENT_REBUILD = 12
ANALYSIS = 20

#
# METHODS
#


def whoCalled(fn):

    def wrapped(*args, **kwargs):
        frames = []
        frame = inspect.currentframe().f_back
        while frame.f_back:
            frames.append(inspect.getframeinfo(frame)[2])
            frame = frame.f_back
        frames.append(inspect.getframeinfo(frame)[2])

        print "===\n{0}\n{1}\n{2}\nCalled by {3}\n____".format(
            fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped


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


class Subprocess(object):

    def __init__(self, proc_type, top_controler, proc=None):
        """Subprocess is a common implementation for the different
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

        self._logger = logging.getLogger("Subprocess Type {0}".format(
            proc_type))

        self._proc_type = proc_type
        self._tc = top_controler
        self._proc = proc
        self._proc_communications = {}
        self._launch_param = None
        self._total = None
        self._exit_code = None
        self._stdin = None
        self._stdout = None
        self._stderr = None
        self._start_time = None
        self._memTestCycle = itertools.cycle(range(10))

    def _send(self, msg, callback, comm_type=None,
              timeout=None, timeout_args=None):

        #Block further communications if it has exited
        if self._exit_code is not None:
            callback(timeout_args)

        timestamp = time.time()
        if timeout is not None:
            timeout += timestamp

        timestamp = "{0:10.10f}".format(timestamp)
        decorated_msg = self._proc.decorate(msg, timestamp)
        self._proc.send(decorated_msg)

        self._proc_communications[timestamp] = (callback, comm_type,
                                                timeout, timeout_args)

    def _handle_callbacks(self, lines):

        timestamp, msg = self._proc.undecorate(lines)

        if timestamp not in self._proc_communications:

            raise BadCommunicateReturn(
                "Unknown communication ({0} {1} not in {2}):\n{3}".format(
                    timestamp, type(timestamp),
                    self._proc_communications.keys(),
                    lines))

        if self._proc_communications[timestamp] is not None:

            callback, comm_type, timeout, timeout_args = (
                self._proc_communications[timestamp])

            del self._proc_communications[timestamp]

            param = self._parse(msg)

            callback(param)

        else:

            self._logger.warning(
                "Communication {0} has allready timed out".format(timestamp))

    def _check_timeouts(self):

        for timestamp in self._proc_communications.keys():

            if self._proc_communications[timestamp] is not None:

                callback, comm_type, timeout, timeout_args = (
                    self._proc_communications[timestamp])

                if timeout is not None and time.time() > timeout:

                    self._logger.info(
                        "Time out callback {0} (msg is {1})".format(
                            callback, timeout_args))

                    if (comm_type == self._proc.PING and
                            self._exit_code is None):

                        self._exit_code = 0

                    callback(timeout_args)

                    self._proc_communications[timestamp] = None

    def _parse(self, msg):

        if self._proc.PING in msg:

            return True

        elif self._proc.INFO in msg:

            self._parse_parameters(msg)
            return self._launch_param

        elif self._proc.PAUSING in msg:

            return True

        elif self._proc.PROGRESS in msg:

            return self._get_val(msg, self._proc.PROGRESS, float)

        elif self._proc.CURRENT in msg:

            return self._get_val(msg, self._proc.CURRENT, int)

        elif self._proc.TOTAL in msg:

            self._total = self._get_val(msg, self._proc.TOTAL, int)
            return self._total

        elif self._proc.REFUSED in msg:

            return True

        elif self._proc.TERMINATING in msg:

            return True

        elif self._proc.UNPAUSING in msg:

            return True

        else:

            self._logger.warning(
                "GUI subprocess got unknown response {0}".format(
                    msg))

    def update(self):

        self._proc.recieve(self._handle_callbacks)
        #self._check_timeouts()

    def get_type(self):
        """Returns the process type"""

        return self._proc_type

    '''
    def isMember(self, methodObject):
        """Checks if methodObject is a member of the class instance

        :param methodObject: The checked object
        :return: If methodObject is a memember of self
        """

        #inspect.getmembers returns (name, value) tuples.
        #the zip makes it ((name1, name2 ...), (value1, value2 ..))
        #and the 1 refers to checking the values, since an object is a value
        return methodObject in zip(*inspect.getmembers(self))[1]
    '''

    def set_callback_is_alive(self, callback):
        """Callback gets allive status"""

        self._send(self._proc.PING, callback,
                   comm_type=self._proc.PING, timeout_args=False)

        if self._memTestCycle.next() == 0:
            self._logger.debug("Current Process Memory Usage: {0:0.1f}%".format(
                self._proc.get_memory_usage()))

    def set_callback_is_paused(self, callback, timeout_args=None):
        """Returns is process is paused"""

        self._send(self._proc.IS_PAUSED, callback, timeout_args=timeout_args)

    def set_callback_parameters(self, callback, timeout_args=None):
        """Returns the parameters used to invoke the process"""

        if self._launch_param is None and self._proc is not None:

            self._send(self._proc.INFO, callback, timeout_args=None)

        else:
            callback(self._launch_param)

    def _param_to_prefix(self, param, callback=None):

        if callback is None:
            callback = self._param_to_prefix_callback

        if callback is None:

            return None

        elif param is not None and 'prefix' in param:
            callback(param['prefix'])
        else:
            callback("")

    def set_callback_prefix(self, callback):
        """Returns the prefix that the project was created with"""

        if self._launch_param is None:
            self._param_to_prefix_callback = callback
            self.set_callback_parameters(self._param_to_prefix)

        else:
            self._param_to_prefix(self._launch_param,
                                  callback=callback)

    def _param_to_init_time(self, param):

        if 'init-time' in self._launch_param:
            self._start_time = self._launch_param['init-time']

    def get_start_time(self):

        if self._start_time is not None:
            return self._start_time

        if self._launch_param is not None:
            self._param_to_init_time(self._launch_param)

        if self._start_time is None:
            return time.time()

        return self._start_time

    def get_exit_code(self):

        if self._exit_code is not None:
            return self._exit_code
        else:
            return 0

    def set_callback_progress(self, callback, timeout_args=None):
        """Returns the progress (as precent).

        Note that this can be different than doing
        proc.get_current()/proc.get_total()
        since the subprocess is free to report a more detailed
        progress than just what iteration step it is on.
        """

        self._send(self._proc.PROGRESS, callback, timeout_args=timeout_args)

    def set_callback_current(self, callback, timeout_args=None):
        """Returns the current iteration step number"""

        self._send(self._proc.CURRENT, callback, timeout_args=timeout_args)

    def set_callback_total(self, callback, timeout_args=None):
        """Returns the total iteration steps"""

        if self._total is None:
            self._send(self._proc.TOTAL, callback, timeout_args=timeout_args)

        else:
            callback(self._total)
        return self._total

    def spawn_proc(self, param_list):

        self.set_logs()

        if self._stdin is None:
            self._stdin = PIPE

        self._logger.info("Launching:\t{0}".format(" ".join(map(str,
                                                                param_list))))

        p = Popen(map(str, param_list), stdin=self._stdin,
                  stdout=self._stdout, stderr=self._stderr, shell=False)

        proc = Proc_IO(self._stdin_path, self._stdout_path,
                       send_file_state='w', trueProcess=p)
        return proc

    def get_sending_path(self):

        return self._stdin_path

    def set_start_time(self):

        self._start_time = time.time()

    def set_process(self, proc):
        """Sets the process, only allowed if not yet set"""

        if self._proc is None:
            self._proc = proc
        else:
            raise AttemptedProcessOverride(self)

    def set_callback_pause(self, callback, timeout_args=None):
        """Requests that the subprocess pauses its operations"""

        self._send(self._proc.PAUSE, callback, timeout_args=timeout_args)

    def set_callback_terminate(self, callback, timeout_args=None):
        """Requests that the subprocess terminates"""

        self._send(self._proc.TERMINATE, callback, timeout_args=timeout_args)

    def set_callback_unpause(self, callback, timeout_args=None):
        """Requests that the subprocess resumes its operations"""

        self._send(self._proc.UNPAUSE, callback, timeout_args=timeout_args)

    def close_communications(self):

        self._proc.close_send_file()
        #self._proc = None

        if hasattr(self._stdin, 'close'):
            self._stdin.close()
            self._stdin = None

        if hasattr(self._stdout, 'close'):
            self._stdout.close()
            self._stdout = None

        if hasattr(self._stderr, 'close'):
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

    def _new_from_paths(self, stdin_path=None, stdout_path=None,
                        stderr_path=None, **params):

        self._set_log('in', stdin_path)
        self._stdout = self._set_log('out', stdout_path, 'r')
        self._stderr = self._set_log('err', stderr_path, 'r')

        proc = Proc_IO(self._stdin_path, self._stdout_path,
                       send_file_state='a')
        return proc

    def _set_log(self, iotype, f_path, iostate1=None, iostate2=None,
                 close_files=False):

        if iotype == 'in':
            self._stdin_path = f_path
        elif iotype == 'out':
            self._stdout_path = f_path
        elif iotype == 'err':
            self._stderr_path = f_path

        fh = None
        if iostate1 is not None:
            fh = open(f_path, iostate1)

            #This will only have an affect if there's no second state
            #and will put marker at EOF to ignore all previous messages
            if iostate1 == 'r':
                fh.read()

        if iostate2 is not None:
            fh.close()
            fh = open(f_path, iostate2)

        if fh is not None and close_files:
            fh.close()
            fh = None

        return fh

    def _parse_parameters(self, params_in_text):

        #FIXIT  rewrite to fit new paradigm
        self._launch_param = {}
        params = self._launch_param

        params_prefix = re.findall(r'__PREFIX__ (.*)', params_in_text)
        if len(params_prefix) > 0:
            params['prefix'] = params_prefix[0]

        params_1pass = re.findall(r'__1-PASS FILE__ (.*)', params_in_text)
        if len(params_1pass) > 0:
            params['1-pass file'] = params_1pass[0]

        params_anal = re.findall(r'__ANALYSIS DIR__ (.*)', params_in_text)
        if len(params_anal) > 0:
            params['analysis-dir'] = params_anal[0]

        params_fixture = re.findall(r'__FIXTURE__ (.*)', params_in_text)
        if len(params_fixture) > 0:
            params['fixture'] = params_fixture[0]

        params_scanner = re.findall(r'__SCANNER__ (.*)', params_in_text)
        if len(params_scanner) > 0:
            params['scanner'] = params_scanner[0]

        params_root = re.findall(r'__ROOT__ (.*)', params_in_text)
        if len(params_root) > 0:
            params['experiments-root'] = params_root[0]

        params_pinning = re.findall(r'__PINNING__ (.*)', params_in_text)
        if len(params_pinning) > 0:
            try:
                params['pinnings-list'] = map(tuple, eval(params_pinning[0]))
            except:
                pass

        params_interval = re.findall(r'__INTERVAL__ (.*)', params_in_text)
        if len(params_interval) > 0:
            params['interval'] = float(params_interval[0])

        params_scans = re.findall(r'__SCANS__ ([0-9]*)', params_in_text)
        if len(params_scans) > 0:
            params['scans'] = int(params_scans[0])

        params_init_time = re.findall(r'__INIT-TIME__ (.*)', params_in_text)
        if len(params_init_time) > 0:
            params['init-time'] = float(params_init_time[0])
        else:
            params['init-time'] = None

        params_cur_image = re.findall(r'__CUR-IM__ ([0-9])', params_in_text)
        if len(params_cur_image) > 0:
            params['current'] = int(params_cur_image[0])
        else:
            params['current'] = None

        if (('interval' in params and 'scan' in params) and
                (params['interval'] is not None and
                 params['scans'] is not None)):

            params['duration'] = params['interval'] * params['scans'] / 60.0


class Experiment_Scanning(Subprocess):

    def __init__(self, top_controller, **params):

        super(Experiment_Scanning, self).__init__(
            EXPERIMENT_SCANNING,
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
        experiment_query['-c'] = sm['experiment-project-id']
        experiment_query['-l'] = sm['experiment-scan-layout-id']
        experiment_query['-u'] = scanner.get_uuid()

        experiment_query['-m'] = _get_pinnings_str(sm['pinnings-list'])

        #self._launch_param = experiment_query

        #Make list of key & value pairs
        e_list += list(itertools.chain.from_iterable(experiment_query.items()))

        return e_list

    def get_proc(self, is_running=False, sm=None, **params):

        if is_running:

            if ('stdin_path' in params and
                    'stdout_path' in params and
                    'stderr_path' in params):

                proc = self._new_from_paths(**params)

            else:

                raise InvalidProcesCreationCall("Can't reconnect without paths")

        else:

            if sm is not None:

                self._new_proc = True
                proc = self.spawn_proc(self.get_param_list(sm))

            else:

                raise InvalidProcesCreationCall(
                    "Cannot create experiment with specific model 'sm'")

        return proc

    def set_logs(self):

        tc = self._tc
        scanner = self._scanner

        self._set_log('in', tc.paths.experiment_stdin.format(
            tc.paths.get_scanner_path_name(scanner.get_name())),
            'w', None, close_files=True)

        """
        if self._new_proc:
            stdin.close()
            stdin = None
        """

        self._stdout = self._set_log(
            'out', tc.paths.log_scanner_out.format(scanner.get_socket()),
            'w', 'r')

        self._stderr = self._set_log(
            'err', tc.paths.log_scanner_err.format(scanner.get_socket()),
            'w', 'r')


class Experiment_Rebuild(Subprocess):

    def __init__(self, top_controller, **params):

        super(Experiment_Rebuild, self).__init__(
            EXPERIMENT_REBUILD,
            top_controller)

        self._new_proc = False
        self._comm_id = None

        self.set_process(self.get_proc(**params))

    def get_proc(self, is_running=False, rebuild_instructions_path=None,
                 comm_id=None, **params):

        if comm_id is None:
            raise InvalidProcesCreationCall(
                "No communications=No way to talk to the process!")
        else:

            self._comm_id = comm_id

        if is_running:

            if ('stdin_path' in params and
                    'stdout_path' in params and
                    'stderr_path' in params):

                proc = self._new_from_paths(**params)

            else:

                raise InvalidProcesCreationCall("Can't reconnect without paths")

        else:

            if rebuild_instructions_path is not None:

                self._new_proc = True
                proc = self.spawn_proc(self.get_param_list(
                    rebuild_instructions_path))

            else:

                raise InvalidProcesCreationCall(
                    "Cannot rebuild experiment without instructiosn path")

        return proc

    def get_comm_id(self):

        return self._comm_id

    def get_param_list(self, rebuild_instructions_path):

        tc = self._tc
        e_list = [tc.paths.make_project]

        #This is a little over the top, but makes it follow same design as
        #the run experiment
        experiment_query = {}
        experiment_query['-i'] = rebuild_instructions_path
        experiment_query['-c'] = self._comm_id

        #Make list of key & value pairs
        e_list += list(itertools.chain.from_iterable(experiment_query.items()))

        return e_list

    def set_logs(self):
        """Sets a new out/err log file pair"""

        paths = self._tc.paths
        self._stdin_path = paths.log_rebuild_in.format(self._comm_id)
        self._stdout_path = paths.log_rebuild_out.format(self._comm_id)
        self._stderr_path = paths.log_rebuild_err.format(self._comm_id)
        self._set_log('in', self._stdin_path, 'w', None, close_files=True)
        self._set_log('out', self._stdout_path, 'w', 'r')
        self._set_log('err', self._stderr_path, 'w', 'r')


class Analysis(Subprocess):

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
            ANALYSIS, top_controller)

        self.set_start_time()
        self._comm_id = None
        self.set_process(self.get_proc(**params))

    def get_param_list(self, comm_id, experiments_root, experiment_prefix,
                       experiment_first_pass, a_dict=None):

        if a_dict is None:
            a_dict = self._tc.config.get_default_analysis_query()

        a_list = [self._tc.paths.analysis]

        #Making the reference to the input file if not supplied in the
        #a_dict.
        if '-i' not in a_dict or a_dict['-i'] in (None, ''):

            if experiment_first_pass is None:

                proc_name = os.path.join(
                    experiments_root,
                    experiment_prefix,
                    self._tc.paths.experiment_first_pass_analysis_relative.format(
                        experiment_prefix))

            else:

                proc_name = experiment_first_pass

            a_dict['-i'] = proc_name

        #Inserting communications id information
        a_dict['-c'] = comm_id

        a_list += list(itertools.chain.from_iterable(a_dict.items()))

        return a_list

    def get_proc(self, is_running=False, a_dict=None,
                 experiments_root=None, experiment_prefix=None,
                 experiment_first_pass=None, comm_id=None):

        if comm_id is None:
            raise InvalidProcesCreationCall(
                "No communications=No way to talk to the process!")
        else:

            self._comm_id = comm_id

        if is_running:
            #Reconnect to running analyssi
            raise InvalidProcesCreationCall("Not able to reconnect yet")

        else:
            #Spawn new process
            if ((experiments_root is not None and
                 experiment_prefix is not None) or (a_dict is not None)):

                param_list = self.get_param_list(comm_id,
                                                 experiments_root,
                                                 experiment_prefix,
                                                 experiment_first_pass,
                                                 a_dict=a_dict)

                proc = self.spawn_proc(param_list)

            else:

                raise InvalidProcesCreationCall(
                    "Missing values for experiments_root or experiment_prefix")

        return proc

    def get_comm_id(self):

        return self._comm_id

    def set_logs(self):
        """Sets a new out/err log file pair"""

        paths = self._tc.paths
        self._stdin_path = paths.log_analysis_in.format(self._comm_id)
        self._stdout_path = paths.log_analysis_out.format(self._comm_id)
        self._stderr_path = paths.log_analysis_err.format(self._comm_id)
        self._set_log('in', self._stdin_path, 'w', None, close_files=True)
        self._set_log('out', self._stdout_path, 'w', 'r')
        self._set_log('err', self._stderr_path, 'w', 'r')
