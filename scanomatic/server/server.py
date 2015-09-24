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
import hashlib

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.app_config as app_config
import scanomatic.server.queue as queue
import scanomatic.server.jobs as jobs
import scanomatic.io.scanner_manager as scanner_manager
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

        self._jobs = jobs.Jobs()
        self._queue = queue.Queue(self._jobs)

        self._scanner_manager = scanner_manager.ScannerPowerManager()

    @property
    def scanner_manager(self):
        """

        :type : scanomatic.io.scanner_manager.ScannerPowerManager
        """
        return self._scanner_manager

    @property
    def queue(self):
        """

        :type self: scanomatic.server.queue.Queue
        """
        return self._queue

    @property
    def jobs(self):
        """

        :type self: scanomatic.server.jobs.Jobs
        """
        return self._jobs

    @property
    def serving(self):

        return self._started

    def shutdown(self):
        self._waitForJobsToTerminate = False
        self._running = False
        return True

    def safe_shutdown(self):
        self._waitForJobsToTerminate = True
        self._running = False
        return True

    def get_server_status(self):

        if self._server_start_time is None:
            run_time = "Not Running"
        else:
            m, s = divmod(time.time() - self._server_start_time, 60)
            h, m = divmod(m, 60)
            run_time = "{0:d}h, {1:d}m, {2:.2f}s".format(
                trunc(h), trunc(m), s)

        return {
            "ServerUpTime": run_time,
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
        sleep = 0.07
        i = 0

        while self._running:

            if i == 0 and self._queue:
                self._attempt_job_creation()
            elif i <= 1:
                self._scanner_manager.update()
            else:
                self._jobs.sync()

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

            self._jobs.force_stop = True

    def _wait_on_jobs(self):
        i = 0
        max_wait_time = 180
        start_time = time.time()
        while self._jobs.running and time.time() - start_time < max_wait_time:

            if i == 0:
                self.logger.info("Waiting for jobs to terminate ({0:.2f}s waiting left)".format(max(
                    0, max_wait_time - (time.time() - start_time))))
            i += 1
            i %= 30
            time.sleep(0.1)

        if self._jobs.running:

            self.logger.warning("Jobs will be abandoned, can't wait for ever..")


    def _get_job_id(self):

        job_id = ""
        bad_name = True

        while bad_name:
            job_id = hashlib.md5(str(time.time())).hexdigest()
            bad_name = job_id in self._queue or job_id in self._jobs

        return job_id

    def get_job(self, job_id):
        """Gets the rpc job model if any corresponding to the id
        :type job_id: str
        :rtype : scanomatic.models.rpc_job_models.RPCjobModel
        """

        if job_id in self._queue:
            return self._queue[job_id]
        return self._jobs[job_id]

    def enqueue(self, model, job_type):

        rpc_job = RPC_Job_Model_Factory.create(
            id=self._get_job_id(),
            pid=None,
            type=job_type,
            status=rpc_job_models.JOB_STATUS.Requested,
            content_model=model)

        if not RPC_Job_Model_Factory.validate(rpc_job):
            self.logger.error("Failed to create job model")
            return False

        if job_type is rpc_job_models.JOB_TYPE.Scan and not self.verify_scanner_claim(rpc_job):
            return False

        self._queue.add(rpc_job)

        self.logger.info("Job {0} with id {1} added to queue".format(rpc_job, rpc_job.id))
        return rpc_job.id

    def verify_scanner_claim(self, rpc_job_model):

        if not self.scanner_manager.has_scanners:
            self.logger.error("There are no scanners reachable from server")
            return False

        return self.scanner_manager.request_claim(rpc_job_model)