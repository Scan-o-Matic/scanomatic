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

    def __init__(self, pipe, loggerName="Pipe effector"):

        self._logger = logger.Logger(loggerName)
        self._pipe = pipe
        self._allowedCalls = dict()

    def setAllowedCalls(self, allowedCalls):
        """Allowed Calls must be iterable with a get item function
        that understands strings"""

        self._allowedCalls = allowedCalls

    def poll(self):

        while self._pipe.poll():

            dataRecvd = self._pipe.recv()

            try:
                response = self._allowedCalls[dataRecvd[0]](*dataRecvd[1],
                                                            **dataRecvd[2])
            except (IndexError, TypeError):

                self._logger.error(
                    "Recieved a malformed data package '{0}'".format(dataRecvd))

            except KeyError:

                self._logger.error("Call to '{0}' not known/allowed".format(
                    dataRecvd[0]))

            else:

                try:
                    if response is not None:
                        self.send(response[0], *response[1], **response[2])

                except:

                    self._logger.error("Could not send response '{0}'".format(
                        response))

    def send(self, callName, *args, **kwargs):

        self._pipe.send((callName, args, kwargs))


class ParentPipeEffector(_PipeEffector):

    def __init__(self, pipe):

        super(ParentPipeEffector, self).__init__(
            pipe, loggerName="Parent Pipe Effector")
        self._status = dict()

        self._allowedCalls['status'] = self._setStatus

    @property
    def status(self):

        #TODO: Modify status to say it is completed if it is
        return self._status

    def _setStatus(self, *args, **kwargs):

        self._status = kwargs


class ChildPipeEffector(_PipeEffector):

    def __init__(self, pipe, procEffector=None):

        if (procEffector is None):
            self._procEffector = None
        else:
            self.procEffector = procEffector

        super(ChildPipeEffector, self).__init__(
            pipe, loggerName="Child Pipe Effector")

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
