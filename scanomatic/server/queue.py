"""The master effector of the analysis, calls and coordinates image analysis
and the output of the process"""
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


#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
import scanomatic.models.rpc_job_models as rpc_job_models

#
# CLASSES
#


class Queue(object):

    def __init__(self):

        self._paths = paths.Paths()
        self._logger = logger.Logger("Job Queue")
        self._next_priority = rpc_job_models.JOB_TYPE.Scan
        self._queue = list(RPC_Job_Model_Factory.serializer.load(self._paths.rpc_queue))

    def __nonzero__(self):

        return len(self._queue) != 0

    def __contains__(self, job_id):

        return any(job.id == job_id for job in self._queue)

    def __getitem__(self, job_id):

        if job_id in self:
            return [job for job in self._queue if job.id == job_id][0]
        return None

    def set_priority(self, job_id, priority):

        job = self[job_id]

        if job:
            job.priority = priority
            RPC_Job_Model_Factory.serializer.dump(job, self._paths.rpc_queue)
            return True
        return False

    def remove(self, job_id):

        job = self[job_id]

        if job:

            self._queue.remove(job)

            return RPC_Job_Model_Factory.serializer.purge(job, self._paths.rpc_queue)

        return False

    def reinstate(self, job):

        if self[job.id] is None:

            self._queue.append(job)
            RPC_Job_Model_Factory.serializer.dump(job, self._paths.rpc_queue)
            return True

        return False

    def get_highest_priority(self):

        job_type = self.__next_priority
        if self.has_job_of_type(job_type):
            return sorted(self.get_job_by_type(job_type), key=lambda job: job.priority)[0]
        return None

    @property
    def __next_priority(self):

        attempts = 0
        while not self.has_job_of_type(self._next_priority) and attempts < len(rpc_job_models.JOB_TYPE):

            next_job_type_value = self._next_priority.value + 1
            if len(rpc_job_models.JOB_TYPE) < next_job_type_value:
                next_job_type_value = 1

            self._next_priority = [job_type for job_type in rpc_job_models.JOB_TYPE
                                   if job_type.value == next_job_type_value][0]
            attempts += 1

        return self._next_priority

    def has_job_of_type(self, job_type):

        return any(*self.get_job_by_type(job_type))

    def get_job_by_type(self, job_type):

        return (job for job in self._queue if job.type == job_type)

    def add(self, job):
        if job.priority < 0:

            if self.has_job_of_type(job.type):
                job.priority = sorted(self.get_job_by_type(job.type),
                                      key=lambda job_in_queue: job_in_queue.priority)[-1] + 1
            else:
                job.priority = 1

        return self.reinstate(job)
