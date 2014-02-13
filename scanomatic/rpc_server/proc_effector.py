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

#
# CLASSES
#


class ProcEffector(object):

    def __init__(self):

        self._specificStatuses = {}
        self._allowedCalls = {}
        self._allowedCalls['start'] = self.start
        self._allowedCalls['stop'] = self.stop
        self._allowedCalls['pause'] = self.pause
        self._allowedCalls['resume'] = self.resume
        self._allowedCalls['status'] = self.status

        self._allowStart = False
        self._running = False
        self._stopping = False
        self._paused = False

        self._gateMessages = False
        self._messages = []

    def start(self):

        if (self._allowStart):
            self._running = True

    def pause(self):

        self._paused = True

    def resume(self):

        self._paused = False

    def stop(self):

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

    def status(self):

        return dict([('running', self._running),
                     ('paused', self._paused),
                     ('stopping', self._stopping),
                     ('messages', self.messages)] +
                    [(k, getattr(self, v)) for k, v in
                     self._specificStatuses.items()])

    @property
    def allowedCalls(self):
        return self._allowedCalls

    def __iter__(self):

        return self

    def next(self):

        while self._running is False and not self._stopping:
            yield


#TO BE MOVED WHEN PROOF OF CONCEPT DONE
class ExperimentEffector(ProcEffector):

    pass
