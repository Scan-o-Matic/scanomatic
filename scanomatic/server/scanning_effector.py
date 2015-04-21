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
from scanomatic.generics import decorators
from scanomatic.models.scanning_model import SCAN_CYCLE, SCAN_STEP


class ScannerEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Scan

    def __init__(self, job):

        # sys.excepthook = support.custom_traceback

        super(ScannerEffector, self).__init__(job, logger_name="Scanner Effector")

        self._specific_statuses['progress'] = 'progress'
        self._specific_statuses['total'] = 'total_images'
        self._specific_statuses['currentImage'] = 'current_image'

        self._allowed_calls['setup'] = self.setup
        self._allowed_calls['add_scanned_image'] = self._add_scanned_image

        self._scanning_job = job.content_model
        self._current_image = -1
        self._previous_scan_time = -1.0
        self._current_step_initiation_time = 0.0
        self._images_ready_for_firstpass_analysis = []
        self._images_requested_scan = []
        self._images_with_started_analysis = []
        self._images_done = []

        self._first_pass_analysis_thread = None
        decorators.register_type_lock(self)

        self._scan_cycle_step = SCAN_CYCLE.Wait

        self._scan_cycle = {
            SCAN_CYCLE.Wait: self._do_wait,
            SCAN_CYCLE.RequestScanner: self._do_request_scanner_on,
            SCAN_CYCLE.RequestScannerOff: self._do_request_scanner_off,
            SCAN_CYCLE.RequestFirstPassAnalysis: self._do_request_first_pass_analysis,
            SCAN_CYCLE.Scan: self._do_scan,
            SCAN_CYCLE.WaitForScanComplete: self._do_wait_for_scan,
            SCAN_CYCLE.WaitForUSB: self._do_wait_for_usb
        }

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

        try:
            step_action = self._scan_cycle[self._scan_cycle_step]()
        except KeyError:
            step_action = self._get_step_to_next_scan_cycle_step()

        self._update_scan_cycle_step(step_action)

        if self._job_completed:
            raise StopIteration
        else:
            return self._scan_cycle_step

    @property
    def _job_completed(self):

        return self._current_image >= self._scanning_job.number_of_scans or self._current_image is None

    def _get_step_to_next_scan_cycle_step(self):

        if self._scan_cycle_step.next_minor is self._scan_cycle_step:
            return SCAN_STEP.NextMajor
        else:
            return SCAN_STEP.NextMinor

    def _update_scan_cycle_step(self, step_action):

        if step_action is SCAN_STEP.NextMajor:
            self._scan_cycle_step = self._scan_cycle_step.next_major
        elif step_action is SCAN_STEP.NextMinor:
            self._scan_cycle_step = self._scan_cycle_step.next_minor

        if not step_action is SCAN_STEP.Wait:
            self._current_step_initiation_time = time.time()

    def _do_wait(self):

        if self._current_image < 0:
            self._start_time = time.time()
            self._previous_scan_time = -self._scanning_job.time_between_scans * 1.1
            self._current_image = 0
            return SCAN_STEP.NextMajor

        project_time = time.time() - self._start_time

        if project_time >= self._scanning_job.time_between_scans:
            self._previous_scan_time = project_time
            return  SCAN_STEP.NextMajor

        return SCAN_STEP.Wait

    def _do_wait_for_usb(self):

        return SCAN_STEP.Wait

    def _do_wait_for_scan(self):

        return SCAN_STEP.Wait

    def _do_request_scanner_on(self):

        return SCAN_STEP.NextMinor

    def _do_request_scanner_off(self):

        return SCAN_STEP.NextMajor

    def _do_request_first_pass_analysis(self):

        return SCAN_STEP.NextMajor

    def _do_scan(self):

        return SCAN_STEP.NextMinor

    def _setup_directory(self):

        os.makedirs(os.path.join(self._scanning_job.directory_containing_project.rstrip(os.sep),
                                 self._scanning_job.project_name))


    @property
    def _firstpass_is_running(self):

        return self._first_pass_analysis_thread is not None and self._first_pass_analysis_thread.isalive()

    @decorators.type_lock
    def _start_first_pass_analysis(self):

        index, time_stamp, path = self._images_ready_for_firstpass_analysis.pop(0)
        self._images_with_started_analysis[index] = (time_stamp, path)

    @decorators.type_lock
    def _add_scanned_image(self, index, time_stamp, path):

        self._images_ready_for_firstpass_analysis.append((index, time_stamp, path))