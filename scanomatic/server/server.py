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

# TODO: Who handles keyboard interrupts?

#
# DEPENDENCIES
#

import time
from math import trunc
import os
import hashlib

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.server.queue as queue
import scanomatic.server.jobs as jobs
import scanomatic.io.scanner_manager as scanner_admin
from scanomatic.io.resource_status import Resource_Status
import scanomatic.generics.decorators as decorators
import scanomatic.models.rpc_job_models as rpc_job_models
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory

#
# CLASSES
#


class Server(object):

    def __init__(self):

        config = app_config.Config()

        self.logger = logger.Logger("Server")
        self.admin = config.rpc_admin
        self._running = False
        self._started = False
        self._waitForJobsToTerminate = False
        self._server_start_time = None

        self._queue = queue.Queue()
        self._jobs = jobs.Jobs()
        self._scanner_manager = scanner_admin.Scanner_Manager()

    @property
    def scanner_manager(self):
        return self._scanner_manager

    @property
    def queue(self):
        return self._queue

    @property
    def jobs(self):
        return self._jobs

    @property
    def serving(self):

        return self._started

    def shutdown(self):
        self._waitForJobsToTerminate = False
        self._running = False
        return  True

    def safe_shutdown(self):
        self._waitForJobsToTerminate = True
        self._running = False
        return  True

    def get_server_status(self):

        if self._server_start_time is None:
            runTime = "Not Running"
        else:
            m, s = divmod(time.time() - self._server_start_time, 60)
            h, m = divmod(m, 60)
            runTime = "{0:d}h, {1:d}m, {2:.2f}s".format(
                trunc(h), trunc(m), s)

        return {
            "ServerUpTime": runTime,
            "QueueLength": len(self._queue),
            "NumberOfJobs": len(self._jobs),
            "ResourceMem": Resource_Status.check_mem(),
            "ResourceCPU": Resource_Status.check_cpu()}

    def start(self):

        if not self._started:
            self._run()
        else:
            self.logger.warning("Attempted to start Scan-o-Matic server that is already running")

    def _attempt_job_creation(self):

        next_job = self._queue.get_highest_priority()
        if next_job is not None:
            if self._jobs.add(next_job):
                self._queue.remove(next_job)

    @decorators.threaded
    def _run(self):

        self._running = True
        self._server_start_time = time.time()
        sleep = 0.51
        i = 0

        while self._running:

            if (i == 0 and self._queue):
                self._attempt_job_creation()
            elif (i <= 1):
                self._scanner_manager.sync()
            else:
                self._jobs.prepare_statuses()

            time.sleep(sleep)
            i += 1
            i %= 3

        self._shutdown_cleanup()

    def _shutdown_cleanup(self):

        self.logger.info("Som Server Main process shutting down")
        self._terminate_jobs()

        if self._waitForJobsToTerminate:
            self._wait_on_jobs()

        self._save_state()

        self.logger.info("Scan-o-Matic server shutdown complete")
        self._started = False

    def _save_state(self):
        pass

    def _terminate_jobs(self):

        if self._jobs.running:
            self.logger.info("Asking all jobs to terminate")

            self._jobs.forceStop = True

    def _wait_on_jobs(self):
        i = 0
        while self._jobs.running:

            if i == 0:
                self.logger.info("Waiting for jobs to terminate")
            i += 1
            i %= 30
            time.sleep(0.1)

    def _get_job_id(self):

        job_id = ""
        bad_name = True

        while bad_name:
            job_id= hashlib.md5(str(time.time())).hexdigest()
            bad_name = job_id in self._queue or job_id in self._jobs

        return job_id

    def get_job(self, job_id):

        return

    def enqueue(self, model, job_type):

        if job_type is rpc_job_models.JOB_TYPE.Scanning and not self.scanner_manager.has_scanners:
            self.logger.error("There are no scanners reachable from server")
            return False

        rpc_job = RPC_Job_Model_Factory.create(
            id=self._get_job_id(),
            pid=os.getpid(),
            type=job_type,
            status=rpc_job_models.JOB_STATUS.Requested,
            contentModel=type(model))

        if not RPC_Job_Model_Factory.validate(rpc_job):
            self.logger.error("Failed to create job model")
            return False

        self._queue.add(rpc_job)

        return rpc_job.id


class SOM_RPC(object):

    def __init__(self):

        self._logger = logger.Logger("RPC Server")
        self._appConfig = app_config.Config()
        self._paths = paths.Paths()

        self._queue = queue.Queue()
        self._jobs = jobs.Jobs()
        self._scannerManager = scanner_admin.Scanner_Manager()


        self._serverStartTime = None
        self._setStatuses([])
        self._server = None
        self._running = False
        self._forceJobsToStop = False
        Resource_Status.loggingLevel('ERROR')

    def _setStatuses(self, statuses, merge=False):

        if merge:
            ids = (s['id'] for s in statuses)
            statuses = [s for s in self._statuses[1:] if s['id'] not in ids] +\
                statuses

        statuses = [self.getServerStatus()] + statuses

        self._statuses = statuses



        return self._statuses


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
