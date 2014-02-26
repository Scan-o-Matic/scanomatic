"""The RPC-server is the master controller of all Scan-o-Matic default
operations.
"""
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

import xmlrpclib
import threading
import time
from SimpleXMLRPCServer import SimpleXMLRPCServer

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.rpc_server.queue as queue
import scanomatic.rpc_server.jobs as jobs
from scanomatic.io.resource_status import Resource_Status

#
# CLASSES
#


class SOM_RPC(object):

    def __init__(self):

        self._logger = logger.Logger("Scan-o-Matic RPC Server")
        self._appConfig = app_config.Config()

        self._paths = paths.Paths()

        self._queue = queue.Queue()
        self._jobs = jobs.Jobs()

        self._admin = self._appConfig.rpc_admin

        self._setStatuses([])
        self._server = None
        self._running = False
        self._forceJobsToStop = False

    def _startServer(self):

        if (self._server is not None):
            raise Exception("Server is already running")

        host = self._appConfig.rpc_host
        port = self._appConfig.rpc_port

        self._server = SimpleXMLRPCServer((host, port), logRequests=False)
        self._server.register_introspection_functions()

        self._running = True
        self._mainThread = None

        self._logger.info("Server listens to {0}:{1}".format(host, port))

        [self._server.register_function(getattr(self, m), m) for m
         in dir(self) if not(m.startswith("_") or m in
                             self._appConfig.rpcHiddenMethods)]

    def _setStatuses(self, statuses):

        #TODO: Extend with some server general info in first slot instead of
        #empty dict
        statuses = [dict()] + statuses

        self._statuses = statuses

    def _main(self):

        sleeps = 30.0

        while self._running:

            sleepDuration = 0.51 * 4

            if (Resource_Status.check_resources()):
                nextJob = self._queue.popHighestPriority()
                if (nextJob is not None):
                    if not self._jobs.add(nextJob):
                        #TODO: should nextJob be recircled or written some
                        #place or added to statuses
                        pass

                    sleepDuration *= 2
                else:
                    sleepDuration *= 20
            elif Resource_Status.currentPasses() == 0:
                sleepDuration *= 15

            self._setStatuses(self._jobs.poll())

            sleepI = 0
            while (self._running and sleepI < sleeps):
                sleepI += 1
                time.sleep(sleepDuration / sleeps)

        self._logger.info("Main process shutting down")
        self._niceQuitProcesses()
        self._logger.info("Shutting down server")
        self._server.shutdown()

    def _niceQuitProcesses(self):

        while self._forceJobsToStop and self._jobs.running:
            self._jobs.forceStop = True
            self._logger.info("Waiting for jobs to terminate")
            time.sleep(0.1)

        self._shutDownComplete = True

    def serverRestart(self, userID):

        if userID == self._admin:
            self.serverShutDown()
            self.run()

    def _serverShutDown(self, forceJobsToStop):
        self._running = False
        self._shutDownComplete = False
        self._forceJobsToStop = forceJobsToStop
        while (self._mainThread is not None and
                self._mainThread.is_alive()):

            time.sleep(0.05)

        self._server = None

    def serverShutDown(self, userID, forceJobsToStop=False):

        if userID == self._admin:
            t = threading.Thread(target=self._serverShutDown,
                                 args=(forceJobsToStop,))
            t.start()
            return True

        return False

    def run(self):

        if self._running is True:

            raise Exception("Server is already running")

        self._logger.info("Starting server")
        self._startServer()

        self._mainThread = threading.Thread(target=self._main)
        self._mainThread.start()

        self._logger.info("Server serves forever")
        self._server.serve_forever()
        self._logger.info("Server Quit")

    def communicateWith(self, userID, jobId, title, kwargs={}):

        if (userID != self._admin):
            return False

        job = self._jobs[jobId]
        if job is not None:
            try:
                job.pipe.send(title, **kwargs)
            except AttributeError:
                self._logger.error("The job {0} has no valid call {1}".format(
                    jobId, title))
                return False
        else:
            self._logger.error("The job {0} is not running".format(jobId))
            return False

        return True

    def getStatuses(self, userID=None):

        return self._statuses

    def getStatus(self, jobId, userID=None):

        return self._jobs.getStatus(jobId)

    def getActiveJobs(self, userID=None):

        return self._jobs.activeJobs

    def getJobsInQueue(self, userID=None):

        return self._queue.getJobsInQueue()

    def createFeatureExtractJob(self, userID, runDirectory, label,
                                priority=None, kwargs={}):

        if userID == self._admin:

            kwargs['runDirectory'] = runDirectory

            self._logger.info("Adding Feature Extraction '{0}' to queue".format(
                label))

            return self._queue.add(
                queue.Queue.TYPE_FEATURE_EXTRACTION,
                jobLabel=label,
                priority=priority,
                **kwargs)
        else:
            self._logger.warning(
                "Unknown user '{0}' tried to create feature extract job".format(
                    userID))
            return False
