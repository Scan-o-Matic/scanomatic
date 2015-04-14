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

import time

#
# INTERNAL DEPENDENCIES
#

import proc_effector
from scanomatic.models.rpc_job_models import JOB_TYPE


class ScannerEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Scan

    def __init__(self, job):

        # sys.excepthook = support.custom_traceback

        super(ScannerEffector, self).__init__(job, logger_name="Scanner Effector")

        self._specific_statuses['progress'] = 'progress'
        self._specific_statuses['total'] = 'total_images'
        self._specific_statuses['currentImage'] = 'current_image'
        self._allowed_calls['setup'] = self.setup
        self._scanning_job = job.content_model
        self._progress = None

    def setup(self, *args, **kwargs):

        pass

    @property
    def progress(self):

        return -1

    @property
    def total_images(self):

        return -1

    @property
    def current_image(self):

        return -1

    def next(self):

        if not self._allow_start:
            return super(ScannerEffector, self).next()

        if self._stopping:
            self._progress = None
            self._running = False

        if self._iteration_index is None:
            self._startTime = time.time()
