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

from ConfigParser import ConfigParser, NoOptionError
from multiprocessing import Pipe

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.paths as paths
import scanomatic.rpc_server.queue as queue
import scanomatic.rpc_server.phenotype_effector as phenotype_effector
import scanomatic.rpc_server.analysis_effector as analysis_effector
import scanomatic.rpc_server.scanner_effector as scanner_effector
import scanomatic.rpc_server.rpc_job as rpc_job
from scanomatic.rpc_server.proc_effector import ProcTypes

#
# CLASSES
#


class Jobs(object):
    def __init__(self):

        self._logger = logger.Logger("Jobs Handler")
        self._paths = paths.Paths()

        self._jobs = {}
        self._jobsData = None
        self._loadFromFile()

        self._forcingStop = False
        self._statuses = []
        self._scanningPids = {}

    @property
    def activeJobs(self):

        return self._jobs.keys()

    @property
    def running(self):

        for job in self._jobs:

            if self._jobs[job].is_alive():
                return True

        return False

    @property
    def forceStop(self):
        return self._forcingStop

    @forceStop.setter
    def forceStop(self, value):

        if value is True:
            self._forcingStop = True
            for job in self._jobs:
                if self._jobs[job].is_alive():
                    self._jobs[job].pipe.send("stop")

        self._forcingStop = value

    @property
    def scanningPids(self):
        return {k: v for k, v in self._scanningPids.items()}

    def __contains__(self, key):

        return key in self._jobs

    def __getitem__(self, key):

        if key in self._jobs:
            return self._jobs[key]
        else:
            self._logger.warning("Unknown job {0} requested".format(key))
            return None

    def _saveJobsData(self):

        self._jobsData.write(open(self._paths.rpc_jobs, 'w'))

    def _loadFromFile(self):

        self._jobsData = ConfigParser(allow_no_value=True)
        try:
            self._jobsData.readfp(open(self._paths.rpc_jobs, 'r'))
        except IOError:
            self._saveJobsData()
            self._logger.info("Created runnig jobs log since non existed")

        for job in self._jobsData.sections():

            childPipe, parentPipe = Pipe()

            try:
                label = self._jobsData.get(job, "label")
            except NoOptionError:
                label = ""
            try:
                pid = int(self._jobsData.get(job, "pid"))
            except (TypeError, NoOptionError):
                pid = -1

            try:
                jobType = ProcTypes.GetByIntRepresentation(
                    int(self._jobsData.get(job, "type")))
            except:
                jobType = ProcTypes.GetDefault()

            self._jobs[job] = rpc_job.Fake(
                job,
                label,
                jobType,
                pid,
                parentPipe)

    def poll(self):

        statuses = []
        jobKeys = self.activeJobs

        for job in jobKeys:

            curJob = self._jobs[job]
            if not self._forcingStop:
                curJob.pipe.poll()
                if curJob.pid < 0:
                    curJob.update_pid()
                    if curJob.pid > 0 and curJob.type == queue.Queue.TYPE_SCAN:
                        self._scanningPids[curJob.identifier] = curJob.pid
                if not curJob.is_alive():
                    del self._jobs[job]
                    self._jobsData.remove_section(job)
                    self._saveJobsData()

            statuses.append(curJob.status)

        self._statuses = statuses
        return statuses

    def _add2JobsData(self, job, setupArgs, setupKwargs):
        """Creates a minimal job data post with sufficient information
        to restart job if need be"""

        self._jobsData.add_section(job.identifier)
        self._jobsData.set(job.identifier, "label", str(job.label))
        self._jobsData.set(job.identifier, "setupArgs", str(setupArgs))
        self._jobsData.set(job.identifier, "setupKwargs", str(setupKwargs))
        self._jobsData.set(job.identifier, "pid", job.pid)
        self._jobs[job.identifier] = job
        self._saveJobsData()

    def fakeProcess(self, jobID, label, job_type, pid):

        childPipe, parentPipe = Pipe()
        job = rpc_job.Fake(jobID, label, job_type, pid, parentPipe)
        self._add2JobsData(job, tuple(), dict())
        return childPipe

    def add(self, procData):
        """Launches and adds a new jobs.
        """

        # VERIFIES NO DUPLICATE IDENTIFIER
        if (procData['id'] in self._jobs or self._jobsData.has_section(
                procData['id'])):
            self._logger.critical(
                "Cannot have jobs with same identifier ({0}), ".format(
                    procData['id']) +
                "new job '{0}' not launched.".format(procData['label']))
            return False

        #SELECTS EFFECTOR BASED ON TYPE
        if (procData["type"] == ProcTypes.EXTRACTION):

            JobEffector = phenotype_effector.PhenotypeExtractionEffector

        elif (procData["type"] == ProcTypes.ANALYSIS):

            JobEffector = analysis_effector.AnalysisEffector

        elif (procData["type"] == ProcTypes.SCANNER):

            JobEffector = scanner_effector.ScannerEffector

        else:

            self._logger.critical(
                ("Job '{0}' ({1}) lost, {2} not yet implemented"
                ).format(procData['label'], procData['id'],
                         procData["type"].textRepresentation))

            return False

        #CONSTRUCTS PIPE PAIR
        parentPipe, childPipe = Pipe()

        #INITIATES JOB EFFECTOR IN TWO STEPS, DON'T REMEMBER WHY
        #identifier, label, target, parentPipe, childPipe
        job = rpc_job.RPC_Job(
            procData['id'],
            procData['label'],
            procData['type'],
            JobEffector,
            parentPipe,
            childPipe)

        job.daemon = True
        job.start()
        job.pipe.send('setup',
                      *procData['args'],
                      **procData['kwargs'])
        job.pipe.send('start')

        #ADDS JOB AND CREATES JOB DATA POST
        self._jobs[job.identifier] = job
        self._add2JobsData(job, procData['args'], procData['kwargs'])

        self._logger.info("Job '{0}' ({1}) started".format(
            job.label, job.identifier))

        return True

    def getStatus(self, jobId):

        statuses = [s for s in self._statuses if s['id'] == jobId]
        return len(statuses) > 0 and statuses[0] or dict()
