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

    def __init__(self, pipe):

        self._logger = logger.Logger("Pipe effector")
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
                if response is not None:
                    self.send(response)

            except:

                self._logger.error("Recieved a malformed data package, " +
                                   "they should have valid string name of " +
                                   "method to run followed by arguments" +
                                   "tuple and keyword arguments dict")

    def send(self, callName, *args, **kwargs):

        self._pipe.send((callName, args, kwargs))


class ParentPipeEffector(_PipeEffector):

    pass


class ChildPipeEffector(_PipeEffector):

    def __init__(self, pipe, procEffector=None):

        if (procEffector is None):
            self._procEffector = None
        else:
            self.procEffector = procEffector

        super(ChildPipeEffector, self).__init__(pipe)

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
