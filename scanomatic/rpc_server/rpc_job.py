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
import scanomatic.io.logger as logger

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

        s = self.pipe.status
        if 'id' not in s:
            s['id'] = self.identifier
        if 'label' not in s:
            s['label'] = self.label
        if 'running' not in s:
            s['running'] = True

        return s

    @property
    def pipe(self):
        return self._parentPipe

    def run(self):

        def _communicator():

            while pipeEffector.keepAlive and jobRunning:
                pipeEffector.poll()
                sleep(0.29)

            _l.info("Will not recieve any more communications")

        jobRunning = True
        _l = logger.Logger("RPC Job")

        pipeEffector = pipes.ChildPipeEffector(
            self._childPipe, self._target(self._identifier, self._label))

        t = Thread(target=_communicator)
        t.start()

        _l.info("Communications thread started")

        effectorIterator = pipeEffector.procEffector

        _l.info("Starting main loop using")

        while t.is_alive() and jobRunning:

            if (pipeEffector.keepAlive):

                try:

                    effectorIterator.next()

                except StopIteration:

                    jobRunning = False
                    #pipeEffector.keepAlive = False

                pipeEffector.sendStatus(pipeEffector.procEffector.status())
                sleep(0.05)

            else:
                sleep(0.29)

        pipeEffector.sendStatus(pipeEffector.procEffector.status())
        t.join()
        _l.info("Job completed")
