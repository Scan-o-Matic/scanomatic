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
import psutil
import setproctitle

#
# INTERNAL DEPENDENCIES
#

import scanomatic.rpc_server.pipes as pipes
import scanomatic.io.logger as logger

#
# CLASSES
#


class Fake(object):

    def __init__(self, identifier, label, jobType, pid, parentPipe):

        self._identifier = identifier
        self._label = label
        self._jobType = jobType
        self._parentPipe = pipes.ParentPipeEffector(parentPipe)
        self._pid = pid
        self._logger = logger.Logger("Fake Process {0}".format(label))
        self._logger.info("Running ({0}) with pid {1}".format(
            self.is_alive(), pid))

    @property
    def type(self):
        return self._jobType

    @property
    def identifier(self):
        return self._identifier

    @property
    def label(self):
        return self._label

    @property
    def pipe(self):
        return self._parentPipe

    @property
    def pid(self):

        return self._pid

    @pid.setter
    def pid(self, value):

        try:
            self._pid = int(value)
        except TypeError:
            self._logger.error("Only ints are valid process IDs ({0})".format(
                value))

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

    def is_alive(self):

        return psutil.pid_exists(self._pid)

    def update_pid(self):

        if self.pid < 0 and 'pid' in self.pipe.status:
            self.pid = self.pipe.status['pid']

class RPC_Job(Process, Fake):

    def __init__(self, identifier, label, jobType, 
                 target, parentPipe, childPipe):

        super(RPC_Job, self).__init__()
        self._label = label
        self._identifier = identifier
        self._target = target
        self._jobType = jobType
        self._parentPipe = pipes.ParentPipeEffector(parentPipe)
        self._childPipe = childPipe
        self._logger = logger.Logger("Job {0} Process".format(label))
        self._pid = -1

    def run(self):

        def _communicator():

            while pipeEffector.keepAlive and jobRunning:
                pipeEffector.poll()
                sleep(0.29)

            _l.info("Will not recieve any more communications")

        jobRunning = True
        _l = logger.Logger("RPC Job (proc-side)")

        pipeEffector = pipes.ChildPipeEffector(
            self._childPipe, self._target(self._identifier, self._label))
        
        setproctitle.setproctitle("SoM {0}".format(
            pipeEffector.procEffector.type))

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
        t.join(timeout=1)
        _l.info("Job completed")
