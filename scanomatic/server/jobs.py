"""Class for launching and gathering subprocesses"""
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

from multiprocessing import Pipe

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
import scanomatic.models.rpc_job_models as rpc_job_models
import scanomatic.server.phenotype_effector as phenotype_effector
import scanomatic.server.analysis_effector as analysis_effector
import scanomatic.server.scanning_effector as scanning_effector
import scanomatic.server.rpcjob as rpc_job
from scanomatic.generics.singleton import SingeltonOneInit
from scanomatic.io import scanner_manager

#
# CLASSES
#


class Jobs(SingeltonOneInit):
    def __one_init__(self):

        self._logger = logger.Logger("Jobs Handler")
        self._paths = paths.Paths()
        self._scanner_manager = scanner_manager.ScannerPowerManager()
        self._jobs = {}
        self._load_from_file()

        self._forcingStop = False
        self._statuses = []

    def __len__(self):
        return len(self._jobs)

    def __contains__(self, key):

        return key in self._jobs

    def __getitem__(self, key):

        if key in self._jobs:
            return self._jobs[key]
        else:
            self._logger.warning("Unknown job {0} requested".format(key))
            return None

    @property
    def active_compile_project_jobs(self):

        return [job for job in self.active_jobs if job.type is rpc_job_models.JOB_TYPE.Compile]

    @property
    def active_jobs(self):

        return self._jobs.keys()

    @property
    def status(self):

        return self._statuses

    @property
    def running(self):

        for job in self._jobs:

            if self._jobs[job].is_alive():
                return True

        return False

    @property
    def force_stop(self):
        return self._forcingStop

    @force_stop.setter
    def force_stop(self, value):

        if value is True:
            self._forcingStop = True
            for job in self._jobs:
                if self._jobs[job].is_alive():
                    self._jobs[job].pipe.send("stop")

        self._forcingStop = value

    def _load_from_file(self):

        jobs = list(RPC_Job_Model_Factory.serializer.load(self._paths.rpc_jobs))
        for job in jobs:
            if job and job.content_model:
                child_pipe, parent_pipe = Pipe()
                self._jobs[job] = rpc_job.Fake(job, parent_pipe)

    def sync(self):

        self._logger.info("Syncing jobs")
        statuses = []
        jobs = self._jobs.keys()
        for job in jobs:
            job_process = self._jobs[job]
            if not self._forcingStop:
                job_process.pipe.poll()
                if not job.pid:
                    job_process.update_pid()
                if not job_process.is_alive():
                    del self._jobs[job]
                    RPC_Job_Model_Factory.serializer.purge(job, self._paths.rpc_jobs)
            statuses.append(job_process.status)

        self.handle_scanners()

        self._statuses = statuses

    def handle_scanners(self):

        self._scanner_manager.update()

        for scanner in self._scanner_manager.non_reported_usbs:
            self._jobs[scanner.owner].pipe.send(scanning_effector.JOBS_CALL_SET_USB, scanner.usb)
            scanner.reported = True
            self._logger.info("Reported USB for scanner {0} to {1}".format(scanner.socket, scanner.owner))

    def add(self, job):
        """Launches and adds a new jobs.
        """
        if any(job.id == j.id for j in self._jobs):
            self._logger.error("Job {0} already exists, will drop current request".format(job.id))
            return True

        job_effector = self._get_job_effector(job)

        # CONSTRUCTS PIPE PAIR
        parent_pipe, child_pipe = Pipe()

        # INITIATES JOB EFFECTOR IN TWO STEPS, DON'T REMEMBER WHY
        # identifier, label, target, parent_pipe, child_pipe

        job_process = rpc_job.RpcJob(
            job,
            job_effector,
            parent_pipe,
            child_pipe)

        self._initialize_job_process(job_process, job)
        self._set_initialized_job(job_process, job)

        self._logger.info("Job '{0}' ({1}) started".format(
            job.id, job.type))

        return True

    def _set_initialized_job(self, job_process, job):

        self._jobs[job] = job_process
        job.status = rpc_job_models.JOB_STATUS.Running
        RPC_Job_Model_Factory.serializer.dump(job, self._paths.rpc_jobs)


    def _initialize_job_process(self, job_process, job):

        """

        :type job_process: scanomatic.server.rpcjob.RpcJob
        """
        job_process.daemon = True
        job_process.start()
        if job.type is rpc_job_models.JOB_TYPE.Scan:
            self._add_scanner_operations_to_job(job_process)
            job.content_model.id = job.id

        job_process.pipe.send('setup', RPC_Job_Model_Factory.serializer.serialize(job))
        job_process.pipe.send('start')

    def _add_scanner_operations_to_job(self, job_process):

        """

        :type job_process: scanomatic.server.rpcjob.RpcJob
        """

        job_process.pipe.add_allowed_calls(self._scanner_manager.subprocess_operations)

    def _get_job_effector(self, job):

        # SELECTS EFFECTOR BASED ON TYPE
        if job.type is rpc_job_models.JOB_TYPE.Features:

            return phenotype_effector.PhenotypeExtractionEffector

        elif job.type is rpc_job_models.JOB_TYPE.Analysis:

            return analysis_effector.AnalysisEffector

        elif job.type is rpc_job_models.JOB_TYPE.Scan:

            return scanning_effector.ScannerEffector

        else:

            self._logger.critical(
                "Job '{0}' ({1}) lost, process not yet implemented".format(job.id, job.type.name))

            return None