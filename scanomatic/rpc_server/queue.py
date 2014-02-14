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

import md5
import ConfigParser
from operator import itemgetter

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.paths as paths
import scanomatic.io.logger as logger

#
# CLASSES
#


class RPC_Subproc_Queue(object):

    TYPE_REBUILD_PROJECT = 0
    TYPE_IMAGE_ANALSYS = 1
    TYPE_FEATURE_EXTRACTION = 2

    def __init__(self):

        self._paths = paths.Paths()
        self._logger = logger.Logger("RPC Subproc Queue")
        self._queue = ConfigParser.ConfigParser()
        try:
            self._queue.readfp(open(self._paths.rpc_queue))
        except IOError:
            self._logger.info("No queue existing, creating new empy")
            self._queue.write(open(self._paths.rpc_queue, 'w'))

    def writeUpdates(self):

        self._queue.write(open(self._paths.rpc_queue, 'w'))

    def setPriority(self, subProcId, priority, writeOnUpdate=True):

        try:
            self._queue.set(subProcId, "priority", priority)
        except ConfigParser.NoSectionError:
            self._logger.error("The subproc id {0} does not exist".format(
                subProcId))
            return False

        if writeOnUpdate:
            self.writeUpdates()

        return True

    def remove(self, subProcId):

        try:
            self._queue.remove_section(subProcId)
        except ConfigParser.NoSectionError:
            self._logger.error("The subproc id {0} is unknown".format(
                subProcId))
            return False

        self.writeUpdates()

        return True

    def getJobInfo(self, jobId):

        try:
            return (jobId, self._queue.get(jobId, "label"),
                    self._queue.getint(jobId, "prio"))
        except:
            self._logger.warning("Problem extracting info on job {0}".format(
                jobId))
            return None

    def getJobsInQueue(self):

        l = [self.getJobInfo(j) for j in self._queue.sections()]
        return sorted([i for i in l if i is not None], key=itemgetter(2),
                      reverse=True)

    def popHighestPriority(self):

        prioSection = None
        highestPrio = -1

        for s in self._queue.sections():

            if not self._queue.has_option(s, "priority"):

                self.setPriority(s, 0, writeOnUpdate=False)

            prio = self._queue.getint(s, "priority")
            if (prio > highestPrio):
                if (prioSection is not None):
                    self.setPriority(prioSection, highestPrio + 1,
                                     writeOnUpdate=False)

                highestPrio = prio
                prioSection = s
            else:
                self.setPriority(prioSection, prio + 1)

        if prioSection is not None:

            procInfo = {'id': prioSection}
            if (self._queue.has_option(prioSection, "label")):
                procInfo['label'] = self._queue.get(prioSection, "label")
            else:
                procInfo['label'] = ""

            if (self._queue.has_option(prioSection, "type"):
                procInfo['type'] = self._queue.getint(prioSection, "type")
            else:
                procInfo['type'] = None

            if (self._queue.has_option(prioSection, "args")):
                procInfo['args'] = self._queue.get(prioSection, "args")
            else:
                procInfo['args'] = ()

            if (self._queue.has_option(prioSection, "kwargs")):
                procInfo['kwargs'] = self._queue.get(prioSection, "kwargs")
            else:
                procInfo['kwargs'] = {}

            self.remove(prioSection)
            self.writeUpdates()
            return procInfo

        return None

    def add(self, subprocType, jobLabel, priority=None, *args, **kwargs):

        if (subprocType not in [self.getattr(p) for p in dir(self) if
                                p.startswith("TYPE_")]):

            self._logger.error("Unknown subprocess type ({0})".format(
                subprocType))
            return None

        if priority is None:
            priority = subprocType

        goodName = False

        while not goodName:
            subprocId = md5.new().hexdigest()
            try:
                self._queue.add_section(subprocId)
                goodName = True
            except ConfigParser.DuplicateSectionError:
                pass

        self._queue.set(subprocId, "type", subprocType)
        self._queue.set(subprocId, "label", jobLabel)

        self.setPriority(subprocId, priority, writeOnUpdate=False)

        self._queue.set(subprocId, "args", args)
        self._queue.set(subprocId, "kwargs", kwargs)

        self.writeUpdates()

        return subprocId
