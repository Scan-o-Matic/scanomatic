import time
import os
from types import StringTypes
import socket
#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.models.rpc_job_models as rpc_job_models
import scanomatic.generics.decorators as decorators
from pipes import ChildPipeEffector
from scanomatic.io import mail
from scanomatic.io.app_config import Config as AppConfig
from threading import Thread

#
# CLASSES
#


class ProcessEffector(object):

    TYPE = rpc_job_models.JOB_TYPE.Unknown

    def __init__(self, job, logger_name="Process Effector"):
        """

        :type job: scanomatic.models.rpc_job_models.RPCjobModel
        :type logger_name: str
        :return:
        """

        self._job = job
        self._job_label = job.id
        self._logger = logger.Logger(logger_name)

        self._fail_vunerable_calls = tuple()

        self._specific_statuses = {}

        self._allowed_calls = {
            'pause': self.pause,
            'resume': self.resume,
            'setup': self.setup,
            'status': self.status,
            'email': self.email,
            'stop': self.stop
        }

        self._allow_start = False
        self._running = False
        self._started = False
        self._stopping = False
        self._paused = False

        self._log_file_path = None

        self._messages = []

        self._iteration_index = None
        self._pid = os.getpid()
        self._pipe_effector = None
        self._start_time = None
        decorators.register_type_lock(self)

    def email(self, add=None, remove=None):

        if add is not None:
            try:
                self._job.content_model.email += [add] if isinstance(add, StringTypes) else add
            except TypeError:
                return False

            return True

        elif remove is not None:

            try:
                self._job.content_model.email.remove(remove)
            except (ValueError, AttributeError, TypeError):
                return False

            return True

        return False

    @property
    def label(self):

        return self._job_label

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
    def progress(self):

        return -1

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

    def setup(self, job):

        self._logger.warning("Setup is not overwritten, job info ({0}) lost.".format(job))

    @property
    def waiting(self):
        if self._stopping:
            return False
        if self._paused or not self._running:
            return True
        return False

    def status(self, *args, **kwargs):

        return dict([('id', self._job.id),
                     ('label', self.label),
                     ('log_file', self._log_file_path),
                     ('pid', self._pid),
                     ('type', self.TYPE.text),
                     ('running', self._running),
                     ('paused', self._paused),
                     ('runTime', self.run_time),
                     ('progress', self.progress),
                     ('stopping', self._stopping),
                     ('messages', self.get_messages())] +
                    [(k, getattr(self, v)) for k, v in
                     self._specific_statuses.items()])

    def stop(self, *args, **kwargs):

        self._stopping = True

    @decorators.type_lock
    def get_messages(self):
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

        if not self._stopping and not self._running:
            if self._allow_start:
                self._running = True
                self._logger.info("Setup passed, switching to run-mode")
                return True
            else:
                self._logger.info("Waiting to run...need setup to allow start")
                return True
        elif self._stopping:
            raise StopIteration

    def _mail(self, title_template, message_template, data_model):

        def _do_mail(title, message, model):

            try:
                if not model.email:
                    self._logger.info(
                        "No mail registered with job, so can't send:\n{0}\n\n{1}".format(
                            title, message
                        ))
                    return

                if AppConfig().mail.server:
                    server = mail.get_server(AppConfig().mail.server, smtp_port=AppConfig().mail.port,
                                             login=AppConfig().mail.user, password=AppConfig().mail.password)
                else:
                    server = None

                if not mail.mail(
                        AppConfig().mail.user,
                        model.email,
                        title.format(**model),
                        message.format(**model),
                        server=server):

                    self._logger.error("Problem getting access to mail server, mail not sent:\n{0}\n\n{1}".format(
                        title.format(**model), message.format(**model)
                    ))
            except KeyError:
                self._logger.error("Malformed message template, model is lacking requested keys:\n{1}\n\n{0}".format(
                    title, message))
            except socket.error:
                self._logger.warning(
                    "Failed to send '{0}' to '{1}'. Probably `sendmail` is not installed. Mail body:\n{2}".format(
                        title.format(**model), model.email, message.format(**model)))

        Thread(target=_do_mail, args=(title_template, message_template, data_model)).start()