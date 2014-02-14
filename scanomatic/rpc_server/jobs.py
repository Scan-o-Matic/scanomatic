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

import ConfigParser

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import scanomatic.io.paths as paths

#
# CLASSES
#

class Jobs(object):

    def __init__(self):

        self._logger = logger.Logger("Running Processes")
        self._paths = paths.Paths()

        self._jobs = {}
        self._jobsData = None
        self._loadFromFile()

        self._forcingStop = False

    @property
    def activeJobs(self):

        return self._jobs.keys()

    @property
    def running(self):

        for job in self._jobs:

            #TODO: This is spoof code:
            if self._jobs[job].status.running is True:
                return True

        return False

    @property
    def forceStop(self):
        return self._forcingStop

    @forcingStop.setter
    def forceStop(self, value):

        if value is True and self._forcingStop is False:

            #TODO: Stop all job
            pass

        self._forcingStop = value

    def __getitem__(self, key):

        if key in self._jobs:
            return self._jobs[key]
        else:
            self._logger.warning("Unknown job {0} requested".format(key))
            return None

    def _loadFromFile(self)

        self._jobsData = ConfigParser(allow_no_value=True)
        try:
            self._jobsData.readfp(open(self._paths.rpc_jobs, 'r'))
        except IOError:
            self._jobsData.write(open(self._paths.rpc_jobs, 'w'))
            self._logger.info("Created runnig jobs log since non existed")

        for job in self._jobsData.sections():

            #TODO: Resume code here

            pass

    def poll(self):

        for job in self._jobs:
            #if self._jobs[job].

            #TODO: Get proc status, record in _jobsData

            pass

    def add(self, procData):
        """
            procData['type']        the process type
            procData['label']       is name
            procData['id']          is call sign
            procData['args']        extra arguments
            procData['kwargs']      extra keyword arguments
        """
        pass

    def getStatus(self, jobId):

        pass
