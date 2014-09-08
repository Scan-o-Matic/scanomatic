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
import sys
import socket

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.rpc_server.queue as queue
import scanomatic.rpc_server.jobs as jobs
import scanomatic.io.scanner_admin as scanner_admin
from scanomatic.io.resource_status import Resource_Status

#
# CLASSES
#

class StoppableXMLRPCServer(SimpleXMLRPCServer):

    def __init__(self, *args, **kwargs):

        SimpleXMLRPCServer.__init__(self, *args, **kwargs)
        self._keepAlive = True

    def stop(self):

        self._keepAlive = False

    def serve_forever(self, poll_interval=0.5):

        while self._keepAlive:
            self.handle_request()
            time.sleep(poll_interval)


class SOM_RPC(object):

    def __init__(self):

        self._logger = logger.Logger("Scan-o-Matic RPC Server")
        self._appConfig = app_config.Config()

        self._paths = paths.Paths()

        self._queue = queue.Queue()
        self._jobs = jobs.Jobs()
        self._scannerManager = scanner_admin.Scanner_Manager()
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

        try:
            self._server = SimpleXMLRPCServer((host, port), logRequests=False)
        except socket.error:
            self._logger.critical(
                "Sever is alread running or the " +
                "port {0} is in use by other program".format(
                    port))

            sys.exit(1)

        self._server.register_introspection_functions()

        self._running = True
        self._mainThread = None

        self._logger.info("Server (pid {0}) listens to {1}:{2}".format(
            os.getpid(), host, port))

        [self._server.register_function(getattr(self, m), m) for m
         in dir(self) if not(m.startswith("_") or m in
                             self._appConfig.rpcHiddenMethods)]

        self._serverStartTime = time.time()

    def _setStatuses(self, statuses, merge=False):

        if merge:
            ids = (s['id'] for s in statuses)
            statuses = [s for s in self._statuses[1:] if s['id'] not in ids] +\
                statuses

        statuses = [self.getServerStatus()] + statuses

        self._statuses = statuses

    def _main(self):

        sleep = 0.51
        i = 0


        while self._running:

            sleepDuration = 0.51 * 4

            if (i == 24 and self._queue):
                if (Resource_Status.check_resources()):
                    nextJob = self._queue.popHighestPriority()
                    if (nextJob is not None):
                        if not self._jobs.add(nextJob):
                            #TODO: should nextJob be recircled or written some
                            #place or added to statuses
                            pass

                        i = 20
                elif (Resource_Status.currentPasses() == 0):
                    i = 10

            if (i % 2):
                self._scannerManager.sync()

            self._setStatuses(self._jobs.poll(), merge=i!=24)

            time.sleep(sleep)
            i += 1
            i %= 25

        self._logger.info("Main process shutting down")
        self._niceQuitProcesses()
        self._logger.info("Shutting down server")
        #self._server.stop()
        self._server.shutdown()

    def _niceQuitProcesses(self):

        self._logger.info("Nice-quitting forcing={0}, jobs running={1}".format(
            self._forceJobsToStop, self._jobs.running))

        i = 0
        while self._forceJobsToStop and self._jobs.running:
            self._jobs.forceStop = True
            if i == 0:
                self._logger.info("Waiting for jobs to terminate")
            i += 1
            i %= 30
            time.sleep(0.1)

        self._shutDownComplete = True

    def _createJobID(self):

        badName = True

        while badName:
            jobID= md5.new(str(time.time())).hexdigest()
            badName = jobID in self._queue or jobID in self._jobs

        return jobID

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

        self._forceJobsToStop = forceJobsToStop
        self._shutDownComplete = False
        self._running = False
        while (self._mainThread is not None and
                self._mainThread.is_alive()):

            time.sleep(0.05)

        self._mainThread = None

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

        self._server.socket.shutdown(socket.SHUT_RDWR)
        self._server.server_close()
        self._server = None
        self._logger.info("Server Quit")
        os._exit(0)

    def reestablishMe(self, userID, jobID, label, pid):
        """Interface for orphaned daemons to re-gain contact with server.

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        jobID : str

            The job identifier of the job that wants to regain contact.
            This job must be known to the server

        label : str

            User-friendly string with info about the job

        pid : int

            The process id of the orphaned daemon

        Returns
        =======

        multiprocessing.Connection or False
            Returns the part of the pipe used by the child-process if
            re-establishment is allowed, else False

        """
        if userID != self._admin:

            self._logger.warning(
                "User '{0}' tried to flush queue".format(
                    userID))

            return False

        if jobID in self._jobs:

            return self._jobs.fakeProcess(jobID, label, pid)

        else:

            self._logger.warning(
                "Unknown job "+
                "'{0}'({1}, pid {2}) tried to claim it exists".format(
                    label, jobID, pid))

            return False

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
                2 --    (str) Job type
                3 --    (int) The job's priority ranking. Jobs are started
                        based on their prioriy. When enqueued they are given
                        a priority based on the job type. Over time, the job
                        accumulates priority such that no job should get
                        stuck in the queue for ever.
        """

        return self._queue.getJobsInQueue()

    def flushQueue(self, userID):
        """Clears the queue

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        Returns
        =======

        bool
            Success status

        See Also
        ========
        
        removeFromQueue
            Remove individual job from queue
        """

        if userID != self._admin:

            self._logger.warning(
                "User '{0}' tried to flush queue".format(
                    userID))

            return False

        while self._queue:
            self._queue.popHighestPriority()

        return True

    def removeFromQueue(self, userID, jobID):
        """Removes job from queue

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        jobID : str
            The ID of job to be removed

        Returns
        =======

        bool
            Success status

        See Also
        ========
        
        flushQueue
            Remove all queued jobs
        """

        if userID != self._admin:

            self._logger.warning(
                "User '{0}' tried to remove job from queue".format(
                    userID))
            return False

        return self._queue.remove(jobID)

    def scanOperations(self, userID, jobID, scanner, operation):
        """Interface for subprocess to request scanner operations

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        jobID : str
            Identifier for the job that owns the scanner to be controlled

        scanner: int or string
            Name of the scanner to be controlled

        operation : str
            "ON" Turn power on to scanner / retrieve USB-port
            "OFF" Turn power off
            "RELEASE" Free scanner from claim
        """

        if userID != self._admin:

            self._logger.warning(
                "User '{0}' tried to manipulate scanners".format(
                    userID))
            return False

        if not scanner in self._scannerManager:

            self._logger.warning(
                "Unknown scanner: {0}".format(scanner))

            return False

        if not self._scannerManager.isOwner(scanner, jobID):

            self._logger.warning(
                "Job '{0}' tried to manipulate someone elses scanners".format(
                    jobID))

            return False

        if operation == "ON":

            usb = self._scannerManager.usb(scanner, jobID)
            if isinstance(usb, str):
                return usb

            return self._scannerManager.requestOn(scanner, jobID)

        elif operation == "OFF":

            return self._scannerManager.requestOff(scanner, jobID)

        elif operation == "RELEASE":

            return self._scannerManager.releaseScanner(scanner)

    def createAnalysisJob(self, userID, inputFile, label,
                          priority=None, kwargs={}):
        """Enques a new analysis job.

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        inputFile : str
            Absolute path to the first pass analysis file containing
            the information about which images to be analysed

        label : str
            Name for the job on the server in human friendly format

        priority : int, optional
            The priority of the job, to put it at the start of the queue

        kwargs : dict, optional
            Further settings to be passed to analysis effector upon setup
            All optional

            configFile: str
                Path to file with setup-instructions.
                Any instructions passed while creating the job will
                overwrite the instructions in the file

            pinningMatrices: list
                List of tuples, one per plate, for pinning formats
                Default: What is specified in first pass file

            localFixture : bool
                If fixture config in directory of the input-file should be
                used.
                Default: False

            lastImage: int
                Index of last image to be included in analysis,
                all images with higher index omitted
                Default: None (all images included)

            outputDirectory: str
                Relative path of output directory for the analysis
                Default: analysis

            watchPosition: ???
                If a specific position should be monitored extra
                Default: None
                
            gridImageIndices: list of int
                The times for which grid-images will be produced
                Default: last image

            supressUnwatched: bool
                If analysis of not watched positions should be omitted
                Default: False

            xmlFormat: dict
                Configuration of xml, known features:
                    'short', 'omit_compartments', 'omit_measures'
                Default: short set to True, no omissions.

            gridArraySettings: dict
                Configuration of grid arryas, known features:
                    'animation'
                Default: animation set to False

            gridSettings: dict
                Configuration of gridding, know features:
                    'use_utso', 'median_coeff', 'manual_threshold'
                Default: Using utso, median coefficient at 0.99 and manual
                threshold at 0.05

            gridCellSettings: dict
                Configuration of grid cells, know features:
                    'blob_detect'
                Default: Use 'default' blob detection

            gridCorrection: list of tuples
                Corrects grids by shifting detected positions by
                indicated amounts
                Default: None

        Returns
        =======

        bool
            Success of enquinig
        """

        if userID != self._admin:

            self._logger.warning(
                "User '{0}' tried to create analysis job".format(
                    userID))
            return False

        if (os.path.abspath(inputFile) != inputFile):

            self._logger.error(
                "Job '{0}' was not started with absolute path".format(
                    label))

            return False

        if (os.path.isfile(inputFile) is False):

            self._logger.error(
                ("Job '{0}' pointed to file ({1}) that doesn't exist".format(
                    label, inputFile)))

            return False
                    
        kwargs['inputFile'] = inputFile

        self._logger.info("Adding Image Analysis {0} based on {1}".format(
            label, inputFile))

        return self._queue.add(
            queue.Queue.TYPE_IMAGE_ANALSYS,
            jobID=self._createJobID(),
            jobLabel=label,
            priority=priority,
            **kwargs)

    def createScanningJob(
            self, userID, scanner, scans, interval, fixture,
            projectsRoot, label, kwargs={}):
        """Attempts to start a scanning job.

        This is a common interface for all type of scan jobs.

        Parameters
        ==========

        userID : str
            The ID of the user requesting to create a job.
            This must match the current ID of the server admin or
            the request will be refused.
            **NOTE**: If using a rpc_client from scanomatic.io the client
            will typically prepend this parameter

        scanner : int or str
            Which scanner to attempt to claim
            **Note**: If scanner is in use, job-creation will fail.

        scans : int
            Number of images to scan
            Must be at least 1

        interval : float
            Number of minutes between scans.
            **NOTE**: Less that 7 minutes is refused

        fixture: str
            Name of fixture to be used.

        projectsRoot : str
            Path to folder in which to place current job

        label: str
            Name of current job and the folder.
            **Note**: If folder exist, job creation will be refused
            **Note**: Only english letters, digits and underscore allowed

        kwargs: dict
            Further settings to override the scanner's defaults or other
            information to be passed 

            Known keys are:

                description: str
                    Information about what is on which plate

                projectID: str
                    Identifier string for the larger project the job is part of

                layoutID: str
                    Identifier string for the current job

                pinning_matrices: list of tuples
                    The number of rows and columns per plate in the fixture.
                    Typically (32, 48) per plate.

                mode: str
                    'TPU' (default) produces transparency scan
                    'COLOR' produces reflective color scan
        """

        if userID != self._admin:

            self._logger.warning(
                "User '{0}' tried to use the scan {1}".format(
                    userID, label))
            return False

        if not isinstance(scans, int) or scans < 1:

            self._logger.error(
                "Invalid number of scans ({0}) for '{1}'".format(
                    scans, label))

            return False

        try:
            interval = float(interval)
        except (ValueError, TypeError):

            self._logger.error(
                "Invalid interval ({0}) for '{1}'".format(
                    interval, label))

            return False

        #TODO: Ask the scanners config/app config instead
        if interval < 7:

            self._logger.error(
                "Interval too short ({0}) for '{1}'".format(
                    interval, label))

            return False

        if self._scannerManager.fixtureExists(fixture) is False:

            self._logger.error(
                "{0}'s fixture '{0}' is not know".format(
                    label, fixture))

            return False

        if not(os.path.isdir(projectsRoot)):

            self._logger.error(
                "{0}'s project root '{0}' is not a directory".format(
                    label, projectsRoot))

            return False

        if len(label) != len(c for c in label
                             if c in string.letter + string.digits + "_"):

            self._logger.error(
                "Label {0} has illegal characters. Only accepting: {1}".format(
                    label, string.letters + string.digits + "_"))

            return False

        if os.path.exists(os.path.join(projectsRoot, label)):

            self._logger.error(
                "{0} already exists in '{0}'".format(
                    label, projectsRoot))

            return False

        jobID = self._createJobID()

        success = self._scannerManager.requestClaim(scanner, os.getpid(), jobID)

        if success:

            #No waiting, lets start
            if not self._jobs.add(dict(
                    id=jobID,
                    type=self._queue.TYPE_SCAN,
                    label=label,
                    args=(scanner, scans, interval, fixture, projectsRoot),
                    kwargs=kwargs
                    )):
                self._scannerManager.releaseScanner(scanner, jobID)
                return False

            return jobID

        return False


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
                jobID=self._createJobID(),
                jobLabel=label,
                priority=priority,
                **kwargs)
        else:
            self._logger.warning(
                "Unknown user '{0}' tried to create feature extract job".format(
                    userID))
            return False
