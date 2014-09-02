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
from math import trunc
import os

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

        self._serverStartTime = None
        self._setStatuses([])
        self._server = None
        self._running = False
        self._forceJobsToStop = False
        Resource_Status.loggingLevel('ERROR')

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

        self._serverStartTime = time.time()

    def _setStatuses(self, statuses):

        statuses = [self.getServerStatus()] + statuses

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

        self._logger.info("Nice-quitting forcing={0}, jobs running={1}".format(
            self._forceJobsToStop, self._jobs.running))

        while self._forceJobsToStop and self._jobs.running:
            self._jobs.forceStop = True
            self._logger.info("Waiting for jobs to terminate")
            time.sleep(0.1)

        self._shutDownComplete = True

    def serverRestart(self, userID, forceJobsToStop=False):
        """This method is not in use since the technical issues have
        not been resolved.

        If implemented, it will restart the server
        """
        #TODO: If this should be possible... how to free the
        #address fron run() ... server_forever()

        return False

        if userID == self._admin:

            t = threading.Thread(target=self._serverRestart,
                                 args=(forceJobsToStop,))
            t.start()
            return True
        return False

    def _serverRestart(self, forceJobsToStop):

        self._serverShutDown(forceJobsToStop)
        self.run()

    def _serverShutDown(self, forceJobsToStop):

        if (self._mainThread is None and self._server is None):
            return

        self._running = False
        self._shutDownComplete = False
        self._forceJobsToStop = forceJobsToStop
        while (self._mainThread is not None and
                self._mainThread.is_alive()):

            time.sleep(0.05)

        self._mainThread = None
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
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            self._logger.info("Server-side forced exit")
            self._serverShutDown(True)
        self._logger.info("Server Quit")

    def communicateWith(self, userID, jobID, title, kwargs={}):
        """Used to communicate with active jobs.

        Args:
            userID (str):   The ID of the user, this must match the
                            current ID of the server admin or request
                            will be refused.

            jobID (str):    The ID for the job to communicate with.

            title (str):    The name, as understood by the job, for what you
                            want to do. The following are universally
                            understood::

                setup:  Passing settings before the job has been started.
                        This ``title`` is preferrably coupled with ``kwargs``
                        while the other universally used titles have no use
                        for extra parameters.
                start:  Starting the job's execution
                pause:  Temporarily pausing the job
                resume: Temporarily resuming the job
                stop:   Stopping the job
                status: Requesting that the job sends back the current status

            kwargs (dict):  Extra parameters to send with the communication.

        Returns:

            bool.   ``True`` if communication was allowed (user was admin and
                    title was accepted imperative) else ``False``
        """

        if (userID != self._admin):
            return False

        job = self._jobs[jobID]
        if job is not None:
            try:
                ret = job.pipe.send(title, **kwargs)
                self._logger.info("The job {0} got message {1}".format(
                    job.identifier, title))
                return ret
            except AttributeError:
                self._logger.error("The job {0} has no valid call {1}".format(
                    job.identifier, title))
                return False
            return True
        else:
            self._logger.error("The job {0} is not running".format(jobID))
            return False

    def getServerStatus(self, userID=None):
        """Gives a dictionary of the servers status

        Kwargs:
            userID (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods

        Returns:
            dictionary. Key value pairs for the different aspects of the
                        server status.
        """

        if self._serverStartTime is None:
            runTime = "Not Running"
        else:
            m, s = divmod(time.time() - self._serverStartTime, 60)
            h, m = divmod(m, 60)
            runTime = "{0:d}h, {1:d}m, {2:.2f}s".format(
                trunc(h), trunc(m), s)
        return {
            "ServerUpTime": runTime,
            "ResourceMem": Resource_Status.check_mem(),
            "ResourceCPU": Resource_Status.check_cpu()}

    def getStatuses(self, userID=None):
        """Gives a list or statuses.

        First entry is always the status of the server followed by
        an item for each job.

        Kwargs:
            userID (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods

        Returns:
            list.   Each item in the list is a dictionary.
                    For information about the job dictionaries and
                    their structure, see ``self.getStatus``.
                    The first item of the list will be a dictionary
                    containing general information about the server.::

            ServerUpTime:  (str) Either the message 'Server Not Running'
                           or a string with like "XXh, YYm, ZZ.ZZs"
                           expressing the time that the server has been
                           running.

        """

        return self._statuses

    def getStatus(self, userID, jobId=None):
        """Gives last recorded status for an active job.

        Args:
            userID (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods
                            (see jobID keyword for more info).

        Kwargs:
            jobID (str):    The job for which to request information.
                            If jobID is not supplied or is None, the
                            userID is assumed to hold the requested
                            job identifier.

        Returns:
            dict            **Either** an empty dictionary if no job
                            with the requested identifier was active
                            **or** a dictionary having as a minimum the
                            following keys::

                id:       (str) The identifier of the job
                label:    (str) The label / job description
                running: (bool) If job is currently running. A job
                            yet to start or having finished would return
                            ``False``
                paused:  (bool) If the job has been paused
                stopping:    (bool) If the job is in the process of
                               being stopped.
                messages:    (list) If the job has messages to
                               communicate with the user.

        .. note::

            The method may return more status information depending on the
            type of job.

            ``RPC_Job.TYPE_FEATURE_EXTRACTION`` also gives these::

                'progress' :    (float) Fraction of total job completed.
                                Not in time, but work-load.
                'runTime' :     (float) Time in seconds that the process
                                have been running

        """

        if jobId is None:
            jobId = userID

        return self._jobs.getStatus(jobId)

    def getActiveJobs(self, userID=None):
        """Gives a list of identifiers for the currently active jobs

        Kwargs:
            userID (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods

        Returns:
            list.   List of active job identifiers
        """
        return self._jobs.activeJobs

    def getJobsInQueue(self, userID=None):
        """Gives a list of enqueued jobs.

        Kwargs:
            userID (str):   The ID of the user requesting status
                            The full purpose of userID is to maintain
                            method interface for all exposed RPC methods

        Returns:
            list.   List of enqueued job each item being a list with the
                    following information (per index).::

                0 --    (str) The job identifier
                1 --    (str) The job label / description
                2 --    (int) The job's priority ranking. Jobs are started
                        based on their prioriy. When enqueued they are given
                        a priority based on the job type. Over time, the job
                        accumulates priority such that no job should get
                        stuck in the queue for ever.
        """

        return self._queue.getJobsInQueue()

    def createFeatureExtractJob(self, userID, runDirectory, label,
                                priority=None, kwargs={}):
        """Enques a new feature extraction job.

        Args:
            userID (str):   The ID of the user, this must match the
                            current ID of the server admin or request
                            will be refused.
            runDirectory (str): The path to the directory containing
                                the native export numpy files from
                                an image analysis job.
                                Note that the path must be absolute.
            label (str):    A free text field for human readable identification
                            of the job.

        Kwargs:
            priority (int): If supplied, the initial priority of the job
                            will not be set by the type of job but by the
                            supplied value.
            kwargs (dict):  Keyword structured arguments to be passed on to
                            the job effector's setup.

        Returns:
            bool.   ``True`` if job request was successfully enqueued, else
                    ``False``
        """

        if userID == self._admin:

            if (not(isinstance(runDirectory, str))):
                self._logger.error(
                    ("Job '{0}' can't be started, " +
                     "invalid runDirectory {1}").format(
                         label, runDirectory))

                return False

            runDirectory = runDirectory.rstrip("/")

            if (os.path.abspath(runDirectory) != runDirectory):

                self._logger.error(
                    "The path for the feature extraction " +
                    "job '{0}' was not absolute path".format(label))

                return False

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
