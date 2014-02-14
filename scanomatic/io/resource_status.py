"""Monitors status of hardware resources"""
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

import psutil

#
# INTERNAL DEPENDENCIES
#

import logger
import app_config

#
# CLASSES
#


class Resource_Status(object):

    _LOGGER = logger.Logger("Hardware Status")
    _APP_CONFIG = app_config.Config()

    _passes = 0

    @staticmethod
    def check_cpu():
        """Checks the CPU status.

        Checks if enough cores (Analysis_Queue.MIN_FREE_CPU_CORES)
        fulfills the usage the criteria
        (Analysis_Queue.MIN_FREE_CPU_PERCENT).

        :returns: boolean
        """

        cur_cpus = psutil.cpu_percent(percpu=True)

        free_cpus = [
            cpu < Resource_Status._APP_CONFIG.resources_cpu_single
            for cpu in cur_cpus]

        Resource_Status._LOGGER.info(
            "CPUs: " + ", ".join(
                ["(({0}%, {1})".format(p, ['FREE', 'TAKEN'][not f]) for
                 p, f in zip(cur_cpus, free_cpus)]))

        cpuOK = (sum(free_cpus) >= Resource_Status._APP_CONFIG.resources_cpu_n
                 and sum(cur_cpus) >
                 Resource_Status._APP_CONFIG.resources_cpu_tot)

        Resource_Status._LOGGER.info(
            "CPUs: {0}".format(['OK', 'NOK'][not cpuOK]))

        return cpuOK

    @staticmethod
    def check_mem():
        """Checks if Phyical Memory status

        Checks if the memory percent usage is below
        Analysis_Queue,MAX_MEM_USAGE.

        :returns: boolean
        """
        memUsage = psutil.phymem_usage().percent
        memOK = memUsage < Resource_Status._APP_CONFIG.resources_mem_min

        Resource_Status._LOGGER.info(
            "MEM: {0}%, {1}".format(
                memUsage, ["OK", "NOT OK"][not memOK]))

        return memOK

    @staticmethod
    def check_resources():
        """Checks if both memory and cpu are OK for poping.

        At least MIN_SUCCESS_PASSES is needed for both checks
        in a row before True is passed

        :returns: boolean
        """

        val = Resource_Status.check_mem() and Resource_Status.check_cpu()
        target = Resource_Status._APP_CONFIG.resources_min_checks

        if val:
            Resource_Status._passes += 1
        else:
            Resource_Status._passes = 0

        Resource_Status._LOGGER.info(
            "System Resource check passed {0}/{1}".format(
                Resource_Status._passes, target))

        ret = Resource_Status._passes >= target

        if ret:
            Resource_Status._passes = 0

        return ret
