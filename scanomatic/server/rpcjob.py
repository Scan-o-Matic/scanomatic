from multiprocessing import Process
from threading import Thread
from time import sleep
import psutil
import os
import setproctitle

#
# INTERNAL DEPENDENCIES
#

import scanomatic.server.pipes as pipes
import scanomatic.io.logger as logger

#
# CLASSES
#


class Fake(object):

    def __init__(self, job, parent_pipe):

        self._job = job
        self._parent_pipe = pipes.ParentPipeEffector(parent_pipe)
        self._logger = logger.Logger("Fake Process {0}".format(job.id))
        self._logger.info("Running ({0}) with pid {1}".format(
            self.is_alive(), job.pid))

        self.abandoned = False

    @property
    def pipe(self):
        """

        :rtype : scanomatic.server.pipes._PipeEffector
        """
        return self._parent_pipe

    @property
    def status(self):

        s = self.pipe.status
        if 'id' not in s:
            s['id'] = self._job.id
        if 'label' not in s:
            s['label'] = self._job.id
        if 'running' not in s:
            s['running'] = True
        if 'progress' not in s:
            s['progress'] = -1
        if 'pid' not in s:
            s['pid'] = os.getpid()

        return s

    def is_alive(self):

        if not self._job.pid:
            return False

        return psutil.pid_exists(self._job.pid)

    def update_pid(self):

        self._job.pid = self.pipe.status['pid']


class RpcJob(Process, Fake):

    def __init__(self, job, job_effector, parent_pipe, child_pipe):

        super(RpcJob, self).__init__()
        self._job = job
        self._job_effector = job_effector
        self._parent_pipe = pipes.ParentPipeEffector(parent_pipe)
        self._childPipe = child_pipe
        self._logger = logger.Logger("Job {0} Process".format(job.id))
        self.abandoned = False

    def run(self):

        def _communicator():

            while pipe_effector.keepAlive and job_running:
                pipe_effector.poll()
                sleep(0.07)

            _l.info("Will not recieve any more communications")

        job_running = True
        _l = logger.Logger("RPC Job {0} (proc-side)".format(self._job.id))

        pipe_effector = pipes.ChildPipeEffector(
            self._childPipe, self._job_effector(self._job))
        
        setproctitle.setproctitle("SoM {0}".format(
            pipe_effector.procEffector.TYPE.name))

        t = Thread(target=_communicator)
        t.start()

        _l.info("Communications thread started")

        effector_iterator = pipe_effector.procEffector

        _l.info("Starting main loop")

        while t.is_alive() and job_running:

            if pipe_effector.keepAlive:

                try:

                    effector_iterator.next()

                except StopIteration:

                    _l.info("Next returned stop iteration, job is done.")
                    job_running = False
                    # pipe_effector.keepAlive = False

                if t.is_alive():
                    pipe_effector.sendStatus(pipe_effector.procEffector.status())
                sleep(0.05)

            else:
                _l.info("Job doesn't want to be kept alive")
                sleep(0.29)

        if t.is_alive():
            pipe_effector.sendStatus(pipe_effector.procEffector.status())
        t.join(timeout=1)
        _l.info("Job completed")
