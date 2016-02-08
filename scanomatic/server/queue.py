import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
import scanomatic.models.rpc_job_models as rpc_job_models
import scanomatic.generics.decorators as decorators
from scanomatic.generics.singleton import SingeltonOneInit
from scanomatic.io.scanner_manager import ScannerPowerManager

#
# CLASSES
#


class Queue(SingeltonOneInit):

    def __one_init__(self, jobs):

        """

        :type jobs: scanomatic.server.jobs.Jobs
        """
        self._paths = paths.Paths()
        self._logger = logger.Logger("Job Queue")
        self._next_priority = rpc_job_models.JOB_TYPE.Scan
        self._queue = list(RPC_Job_Model_Factory.serializer.load(self._paths.rpc_queue))
        self._scanner_manager = ScannerPowerManager()
        self._jobs = jobs
        decorators.register_type_lock(self)

    @decorators.type_lock
    def __len__(self):

        return len(self._queue)

    @decorators.type_lock
    def __nonzero__(self):

        return len(self._queue) != 0

    @decorators.type_lock
    def __contains__(self, job_id):

        return any(job.id == job_id for job in self._queue)

    @decorators.type_lock
    def __getitem__(self, job_id):

        if job_id in self:
            return [job for job in self._queue if job.id == job_id][0]
        return None

    @property
    @decorators.type_lock
    def status(self):
        return [RPC_Job_Model_Factory.to_dict(m) for m in self._queue]

    @decorators.type_lock
    def set_priority(self, job_id, priority):

        job = self[job_id]

        if job:
            job.priority = priority
            RPC_Job_Model_Factory.serializer.dump(job, self._paths.rpc_queue)
            return True
        return False

    @decorators.type_lock
    def remove(self, job):

        if job.id in self:

            self._logger.info("Removing job {0} from queue".format(job.id))
            self._queue.remove(job)
            return RPC_Job_Model_Factory.serializer.purge(job, self._paths.rpc_queue)

        self._logger.warning("No known job {0} in queue, can't remove".format(job.id))
        return False

    @decorators.type_lock
    def remove_and_free_potential_scanner_claim(self, job):

        if self.remove(job):
            if job.type is rpc_job_models.JOB_TYPE.Scan:
                return self._scanner_manager.release_scanner(job.id)
            return True
        return False

    @decorators.type_lock
    def reinstate(self, job):

        if self[job.id] is None:
            job.status = rpc_job_models.JOB_STATUS.Queued
            self._queue.append(job)
            RPC_Job_Model_Factory.serializer.dump(job, self._paths.rpc_queue)
            return True

        return False

    @decorators.type_lock
    def get_highest_priority(self):

        job_type = self.__next_priority_job_type
        if self._has_job_of_type(job_type):
            jobs = list(self._get_job_by_type(job_type))
            if job_type is rpc_job_models.JOB_TYPE.Compile:
                jobs = self._get_allowed_compile_project_jobs(jobs)

            ordered_jobs = sorted(jobs, key=lambda job: job.priority)
            self._logger.info("The available jobs of type {0} are queued as {1}".format(
                job_type, [(job, job.id) for job in ordered_jobs]))
            if ordered_jobs:
                return ordered_jobs[0]
        else:
            self._logger.info("No {0} jobs".format(job_type))
        return None

    def _get_allowed_compile_project_jobs(self, queued_jobs):

        def _start_condition_met(this_job):
            if this_job.content_model.start_condition in self:
                return False

            for active_job in self._jobs.active_compile_project_jobs:

                if active_job.id == this_job.content_model.start_condition:

                    self._logger.info("Can't launch job {0} becase of job {1}".format(
                        (this_job, this_job.id), (active_job, active_job.id)))
                    return False

            return True

        return [job for job in queued_jobs if _start_condition_met(job)]

    @property
    def __next_priority_job_type(self):

        attempts = 0
        while not self._has_job_of_type(self._next_priority) and attempts < len(rpc_job_models.JOB_TYPE):

            self._next_priority = self._next_priority.cycle
            attempts += 1

        return self._next_priority

    def _has_job_of_type(self, job_type):

        return any(self._get_job_by_type(job_type))

    def _get_job_by_type(self, job_type):

        return (job for job in self._queue if job.type == job_type)

    def add(self, job):

        if job.priority < 0:

            if self._has_job_of_type(job.type):
                job.priority = sorted(self._get_job_by_type(job.type),
                                      key=lambda job_in_queue: job_in_queue.priority)[-1].priority + 1
            else:
                job.priority = 1

        return self.reinstate(job)
