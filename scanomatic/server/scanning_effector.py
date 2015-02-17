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

import os
from ConfigParser import ConfigParser
import time

#
# INTERNAL DEPENDENCIES
#

import proc_effector
import scanomatic.io.logger as logger
from scanomatic.server.proc_effector import ProcTypes

class ScannerEffector(proc_effector.ProcEffector):

    TYPE = ProcTypes.SCANNER 

    def __init__(self, identifier, label):

        #sys.excepthook = support.custom_traceback

        super(ScannerEffector, self).__init__(identifier, label,
                                               loggerName="Scanner Effector")

        self._specificStatuses['progress'] = 'progress'
        self._specificStatuses['total'] = 'totalImages'
        self._specificStatuses['currentImage'] = 'curImage'

        self._allowedCalls['setup'] = self.setup

    def setup(self, *lostArgs, **scannerKwargs):

        pass

    def next(self):

        if not self._allowStart:
            return super(ScannerEffector, self).next()

        if self._stopping:
            self._progress = None
            self._running = False

        if self._iteratorI is None:
            self._startTime = time.time()
