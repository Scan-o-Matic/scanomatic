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
import os

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
        self._current_image = -1
        self._previous_scan_time = -1.0

    def setup(self, *args, **kwargs):

        self._setup_directory()
        self._allow_start = True

    @property
    def progress(self):

        if self._current_image < 0:
            return 0.0
        elif self._current_image is None:
            return 1.0
        else:
            return float(self._current_image + 1.0) / self.total_images

    @property
    def total_images(self):

        return self._scanning_job.number_of_scans

    @property
    def current_image(self):

        return self._current_image

    def next(self):

        if not self._allow_start:
            return super(ScannerEffector, self).next()

        if self._current_image < 0:
            self._start_time = time.time()
            self._previous_scan_time = -self._scanning_job.time_between_scans * 1.1
            self._current_image = 0

        project_time = time.time() - self._start_time

        if project_time >= self._scanning_job.time_between_scans:
            self._previous_scan_time = project_time
            # Request scanner

        # if can scan, scan

        # if can analyse start analysis

        if self._current_image >= self._scanning_job.number_of_scans or self._current_image is None:
            raise StopIteration

    def _setup_directory(self):

        os.makedirs(os.path.join(self._scanning_job.directory_containing_project.rstrip(os.sep),
                                 self._scanning_job.project_name))