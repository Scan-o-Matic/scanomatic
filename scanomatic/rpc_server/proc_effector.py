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

from time import sleep
import os

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger

#
# CLASSES
#


class ProcEffector(object):

    TYPE = -1

    def __init__(self, identifier, label, loggerName="Process Effector"):

        self._identifier = identifier
        self._label = label
        self._logger = logger.Logger(loggerName)
        self._type = "Generic"

        self._failVunerableCalls = tuple()

        self._specificStatuses = {}
        self._allowedCalls = {}
        self._allowedCalls['pause'] = self.pause
        self._allowedCalls['resume'] = self.resume
        self._allowedCalls['setup'] = self.setup
        self._allowedCalls['start'] = self.start
        self._allowedCalls['status'] = self.status
        self._allowedCalls['stop'] = self.stop
        
        self._allowStart = False
        self._running = False
        self._started = False
        self._stopping = False
        self._paused = False

        self._gateMessages = False
        self._messages = []

        self._iteratorI = None
        self._pid = os.getpid()

    @property
    def type(self):
        return self._type

    @property
    def failVunerableCalls(self):

        return self._failVunerableCalls

    @property
    def keepAlive(self):

        return not self._started and not self._stopping or self._running

    @property
    def identifier(self):
        return self._identifier

    @property
    def label(self):
        return self._label

    def pause(self, *args, **kwargs):

        self._paused = True

    def resume(self, *args, **kwargs):

        self._paused = False

    def setup(self, *args, **kwargs):

        if (len(args) > 0 or len(kwargs) > 0):
            self._logger.warning(
                "Setup is not overwritten, {0} and {1} lost.".format(
                    args, kwargs))

    def start(self, *args, **kwargs):

        if (self._allowStart):
            self._running = True

    def status(self, *args, **kwargs):

        return dict([('id', self._identifier),
                     ('pid', self._pid),
                     ('label', self._label),
                     ('type', self._type),
                     ('running', self._running),
                     ('paused', self._paused),
                     ('stopping', self._stopping),
                     ('messages', self.messages)] +
                    [(k, getattr(self, v)) for k, v in
                     self._specificStatuses.items()])

    def stop(self, *args, **kwargs):

        self._stopping = True

    def _messageGate(self):
        while self._gateMessages:
            sleep(0.07)
        self._gateMessages = True

    @property
    def messages(self):

        self._messageGate()
        msgs = self._messages
        self._messages = []
        self._gateMessages = False
        return msgs

    def addMessage(self, msg):

        self._messageGate()
        self._messages.append(msg)
        self._gateMessages = False

    @property
    def allowedCalls(self):
        return self._allowedCalls

    def __iter__(self):

        return self

    def next(self):

        while self._running is False and not self._stopping:
            sleep(0.1)
            self._logger.debug(
                "Pre-running and waiting run {0} and stop {1}".format(
                    self._running, self._stopping))
            return None

        if self._stopping:
            raise StopIteration
