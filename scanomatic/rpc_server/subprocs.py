"""Extensions of the multiprocessings Process for ease of use"""
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

from multiprocessing import Process
from threading import Thread
from time import sleep

#
# CLASSES
#


class RPC_Proc(Process):

    def __init__(self, pipeEffector, ProcEffector):

        super(RPC_Proc, self).__init__()
        self.pipeEffector = pipeEffector
        self.pipeEffector.procEffector = ProcEffector()
        self._keepRunning = True

    def _communicator(self):

        while self._keepRunning:
            self.pipeEffector.poll()
            sleep(0.29)

    def run(self):

        t = Thread(target=self._communicator)
        t.start()

        while t.isalive:

            if (self._keepRunning):

                try:

                    self.pipEffector.procEffector.next()

                except StopIteration:

                    self._keepRunning = False


class ProcEffector(object):

    pass


class AnalysisEffector(ProcEffector):

    pass


class ExperimentEffector(ProcEffector):

    pass


class FeaturesEffector(ProcEffector):

    pass
