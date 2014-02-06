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


class PipeEffector(object):

    def __init__(self, pipe):

        self._logger = logger.Logger("Pipe effector")
        self._pipe = pipe
        self._allowedCalls = dict()
        self._procEffector = None

    def poll(self):

        while self._pipe.poll():

            dataRecvd = self._pipe.recv()

            try:
                response = self._allowedCalls[dataRecvd[0]](**dataRecvd[1])
                if response is not None:
                    self.send(response)

            except:

                self._logger.error("Recieved a malformed data package, " +
                                   "they should have valid string name of " +
                                   "method to run and a dict of keyword " +
                                   "arguments")

    def send(self, status):

        self._pipe.send(status)

    @property
    def procEffector(self):

        return self._procEffector

    @procEffector.setter
    def procEffector(self, procEffector):

        self._procEffector = procEffector
        self._allowedCalls = procEffector.allowedCalls
