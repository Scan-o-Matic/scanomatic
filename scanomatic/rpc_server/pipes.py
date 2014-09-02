"""Factory for pipe effectors"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger

#
# CLASSES
#


class _PipeEffector(object):

    REQUEST_ALLOWED = "__ALLOWED_CALLS__"

    def __init__(self, pipe, loggerName="Pipe effector"):

        self._logger = logger.Logger(loggerName)
        self._pipe = pipe
        self._allowedCalls = dict()
        self._allowedRemoteCalls = None

    def setAllowedCalls(self, allowedCalls):
        """Allowed Calls must be iterable with a get item function
        that understands strings"""

        self._allowedCalls = allowedCalls
        self._sendOwnAllowedKeys()

    def _sendOwnAllowedKeys(self):

        self._logger.info("Informing other side of pipe about my allowed calls")
        self.send(self.REQUEST_ALLOWED, *self._allowedCalls.keys())

    def poll(self):

        while self._pipe.poll():

            response = None
            dataRecvd = self._pipe.recv()
            self._logger.debug("Pipe recieved {0}".format(dataRecvd))

            try:
                if dataRecvd[0] == self.REQUEST_ALLOWED:
                    self._logger.info(
                        "Got information about other side's allowed " +
                        "calls '{0}'".format(
                            dataRecvd[1]))
                    if self._allowedRemoteCalls is None:
                        self._sendOwnAllowedKeys()
                    self._allowedRemoteCalls = dataRecvd[1]
                    response = None
                else:
                    if self._allowedRemoteCalls is None:
                        self._logger.info(
                            "No allowed calls")
                        self._sendOwnAllowedKeys()

                    response = self._allowedCalls[dataRecvd[0]](*dataRecvd[1],
                                                                **dataRecvd[2])
            except (IndexError, TypeError):

                self._logger.error(
                    "Recieved a malformed data package '{0}'".format(dataRecvd))

            except KeyError:

                self._logger.error("Call to '{0}' not known/allowed".format(
                    dataRecvd[0]))
                self._logger.info("Allowed calls are '{0}'".format(
                    self._allowedCalls.keys()))

            else:

                """
                self._logger.critical("Unforseen error calling '{0}'".format(
                    dataRecvd))
                """
                try:
                    if response is not None:
                        if (isinstance(response, dict) and
                                dataRecvd[0] == "status"):
                            self.send(dataRecvd[0], **response)
                            self._logger.info("Sent status response {0}".format(
                                response))
                        else:
                            self.send(response[0], *response[1], **response[2])
                            self._logger.info("Sent response {0}".format(
                                response))

                    else:
                        self._logger.debug("No response")

                except:

                    self._logger.error("Could not send response '{0}'".format(
                        response))

    def send(self, callName, *args, **kwargs):

        if (self._allowedRemoteCalls is None or
                callName == self.REQUEST_ALLOWED or
                callName in self._allowedRemoteCalls):

            self._pipe.send((callName, args, kwargs))
            return True

        else:
            self._logger.warning("Other side won't accept '{0}'. ".format(
                callName) + "Known calls are '{0}'".format(
                    self._allowedRemoteCalls))
            return False


class ParentPipeEffector(_PipeEffector):

    def __init__(self, pipe):

        super(ParentPipeEffector, self).__init__(
            pipe, loggerName="Parent Pipe Effector")
        self._status = dict()

        self._allowedCalls['status'] = self._setStatus

    @property
    def status(self):

        return self._status

    def _setStatus(self, *args, **kwargs):

        #self._logger.info("Pipe made its status {0}".format(kwargs))
        self._status = kwargs


class ChildPipeEffector(_PipeEffector):

    def __init__(self, pipe, procEffector=None):

        super(ChildPipeEffector, self).__init__(
            pipe, loggerName="Child Pipe Effector")

        if (procEffector is None):
            self._procEffector = None
            self._allowedCalls = {}
        else:
            self.procEffector = procEffector

    @property
    def keepAlive(self):

        return (self._procEffector is None and True or
                self._procEffector.keepAlive)

    @property
    def procEffector(self):

        return self._procEffector

    @procEffector.setter
    def procEffector(self, procEffector):

        self._procEffector = procEffector
        self.setAllowedCalls(procEffector.allowedCalls)

    def sendStatus(self, status):

        self.send('status', **status)
