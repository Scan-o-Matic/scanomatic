#!/usr/bin/env python
"""The Analysis Queue"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

from collections import deque
import psutil

#
# INTERNAL DEPENDENCIES
#

from subproc_collection_interface import SubProc_Collection_Interface

#
# CLASSES
#


class Analysis_Queue(SubProc_Collection_Interface):

    MAX_MEM_USAGE = 60
    MIN_FREE_CPU_PERCENT = 50
    MIN_FREE_CPU_CORES = 2

    def __init__(self):
        """Analysis Queue extends the collections.deque.

        It exposes 3 methods

        Pop
        ====

        Analysis_Queue.pop() pops from the left side
        of the queue, making it follow FIFO (First In -
        First Out) logic.

        The pop method is restriced such that pop
        will only yeild an element if CPU and
        physical memory usage fullfills certain
        conditions.

        Conditions
        ----------

        CPU
        ...

        Current CPU usage is checked per core so that
        there are a minimum of Analysis_Queue.MIN_FREE_CPU_CORES
        (Class Static Variable) with at least
        Analysis_Queue.MIN_FREE_CPU_PERCENT (Class Static
        Variable) free resources.

        Physical Memory
        ...............

        Current physical memory percent usage is compared
        to Analysis_Queue.MAX_MEM_USAGE (Class Static Variable).


        Count
        -----

        Reports the number of elements currently in the
        queue.

        Push
        ----

        Push adds a new element to the right side of
        the queue.
        """

        self._queue = deque()

    def __iter__(self):
        """Obtaining iter for the queue"""
        return self._queue.__iter__()

    def _check_cpu(self):
        """Checks the CPU status.

        Checks if enough cores (Analysis_Queue.MIN_FREE_CPU_CORES)
        fulfills the usage the criteria
        (Analysis_Queue.MIN_FREE_CPU_PERCENT).

        :returns: boolean
        """

        cur_cpu = psutil.cpu_percent(percpu=True)
        free_cpu = [cpu < self.MIN_FREE_CPU_PERCENT for cpu in cur_cpu]
        return len(free_cpu) >= self.MIN_FREE_CPU_CORES

    def _check_mem(self):
        """Checks if Phyical Memory status

        Checks if the memory percent usage is below
        Analysis_Queue,MAX_MEM_USAGE.

        :returns: boolean
        """

        return psutil.phymem_usage().percent < self.MAX_MEM_USAGE

    def _check_resources(self):
        """Checks if both memory and cpu are OK for poping.

        :returns: boolean
        """

        return self._check_mem() and self._check_cpu()

    def count(self):
        """Checks the length of the queue.

        :returns: integer
        """

        return len(self._queue)

    def pop(self):
        """Checks if memory and cpu criteria is met and pops element.

        If no element is present, or may be returned, None is passed.

        :returns: Queue Element or None
        """

        if len(self._queue) > 0 and self._check_resources():

            #Return settings for next analysis since
            #there are resources
            return self._queue.popleft()

        else:

            return None

    def push(self, elem):
        """Puts a queue element at the end of the queue.

        :param elem: A queue element
        :returns: True
        """

        self._queue.append(elem)
        return True
