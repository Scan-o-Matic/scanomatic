"""Extensions of the multiprocessings Process for ease of use"""
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

from multiprocessing import Process
from threading import Thread
from time import sleep

#
# INTERNAL DEPENDENCIES
#

import scanomatic.rpc_server.pipes as pipes

#
# CLASSES
#


class RPC_Job(Process):

    def __init__(self, identifier, label, target, parentPipe, childPipe):

        super(RPC_Job, self).__init__()
        self._label = label
        self._identifier = identifier
        self._target = target
        self._parentPipe = pipes.ParentPipeEffector(parentPipe)
        self._childPipe = childPipe

    @property
    def identifier(self):
        return self._identifier

    @property
    def label(self):
        return self._label

    @property
    def status(self):

        return self.pipe.status

    @property
    def pipe(self):
        return self._parentPipe

    def run(self):

        def _communicator():

            while pipeEffector.keepAlive:
                pipeEffector.poll()
                sleep(0.29)

        pipeEffector = pipes.ChildPipeEffector(
            self._childPipe, self._target(self._identifier, self._label))

        t = Thread(target=_communicator)
        t.start()

        while t.is_alive():

            if (pipeEffector.keepAlive):

                try:

                    pipeEffector.procEffector.next()

                except StopIteration:

                    pass
                    #pipeEffector.keepAlive = False
            else:
                sleep(0.29)
