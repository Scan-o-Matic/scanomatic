"""Abstract class for all proc effectors"""
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
import os

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.models.rpc_job_models as rpc_job_models
import scanomatic.generics.decorators as decorators
from pipes import ChildPipeEffector

#
# CLASSES
#


class ProcessEffector(object):

    TYPE = rpc_job_models.JOB_TYPE.Unknown

    def __init__(self, job, logger_name="Process Effector"):

        self._job = job
        self._logger = logger.Logger(logger_name)

        self._fail_vunerable_calls = tuple()

        self._specific_statuses = {}

        self._allowed_calls = {
            'pause': self.pause,
            'resume': self.resume,
            'setup': self.setup,
            'start': self.start,
            'status': self.status,
            'stop': self.stop
        }

        self._allow_start = False
        self._running = False
        self._started = False
        self._stopping = False
        self._paused = False

        self._messages = []

        self._iteration_index = None
        self._pid = os.getpid()
        self._pipe_effector = None
        self._start_time = None
        decorators.register_type_lock(self)

    @property
    def pipe_effector(self):
    
        """

        :rtype : ChildPipeEffector
        """
        if self._pipe_effector is None:
            self._logger.warning("Attempting to get pipe effector that has not been set")
            
        return self._pipe_effector
    
    @pipe_effector.setter
    def pipe_effector(self, value):
        
        """

        :type value: ChildPipeEffector
        """
        if value is None or isinstance(value, ChildPipeEffector):
            self._pipe_effector = value
    
    @property
    def run_time(self):

        if self._start_time is None:
            return 0
        else:
            return time.time() - self._start_time

    @property
    def fail_vunerable_calls(self):

        return self._fail_vunerable_calls

    @property
    def keep_alive(self):

        return not self._started and not self._stopping or self._running

    def pause(self, *args, **kwargs):

        self._paused = True

    def resume(self, *args, **kwargs):

        self._paused = False

    def setup(self, *args, **kwargs):

        if 0 < len(args) or 0 < len(kwargs):
            self._logger.warning(
                "Setup is not overwritten, {0} and {1} lost.".format(
                    args, kwargs))

    def start(self, *args, **kwargs):

        if self._allow_start:
            self._running = True

    def status(self, *args, **kwargs):

        return dict([('id', self._job.id),
                     ('pid', self._pid),
                     ('type', self.TYPE.text),
                     ('running', self._running),
                     ('paused', self._paused),
                     ('runTime', self.run_time),
                     ('stopping', self._stopping),
                     ('messages', self.messages)] +
                    [(k, getattr(self, v)) for k, v in
                     self._specific_statuses.items()])

    def stop(self, *args, **kwargs):

        self._stopping = True

    @decorators.type_lock
    @property
    def messages(self):
        msgs = self._messages
        self._messages = []
        return msgs

    @decorators.type_lock
    def add_message(self, msg):
        self._messages.append(msg)

    @property
    def allow_calls(self):
        return self._allowed_calls

    def __iter__(self):

        return self

    def next(self):

        while self._running is False and not self._stopping:
            time.sleep(0.1)
            self._logger.debug(
                "Pre-running and waiting run {0} and stop {1}".format(
                    self._running, self._stopping))
            return None

        if self._stopping:
            raise StopIteration
