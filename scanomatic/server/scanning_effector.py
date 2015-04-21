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
from scanomatic.models.scanning_model import SCAN_CYCLE, SCAN_STEP, ScanningModelEffectorData


class ScannerEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Scan
    WAIT_FOR_USB_TOLERANCE_FACTOR = 0.33
    WAIT_FOR_SCAN_TOLERANCE_FACTOR = 0.5

    def __init__(self, job):

        # sys.excepthook = support.custom_traceback

        super(ScannerEffector, self).__init__(job, logger_name="Scanner Effector")

        self._specific_statuses['progress'] = 'progress'
        self._specific_statuses['total'] = 'total_images'
        self._specific_statuses['currentImage'] = 'current_image'

        self._allowed_calls['setup'] = self.setup
        self._allowed_calls['set_usb'] = self._set_usb_port

        self._scanning_job = job.content_model
        self._scanning_effector_data = ScanningModelEffectorData()

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

        if self.current_image < 0:
            return 0.0
        elif self.current_image is None:
            return 1.0
        else:
            return float(self.current_image + 1.0) / self.total_images

    @property
    def total_images(self):

        return self._scanning_job.number_of_scans

    @property
    def current_image(self):

        return self._scanning_effector_data.current_image

    def next(self):

        if not self._allow_start:
            return super(ScannerEffector, self).next()

        try:
            step_action = self._scan_cycle[self._scanning_effector_data.current_cycle_step]()
        except KeyError:
            step_action = self._get_step_to_next_scan_cycle_step()

        self._update_scan_cycle_step(step_action)

        if self._job_completed:
            raise StopIteration
        else:
            return self._scan_cycle_step

    @property
    def _job_completed(self):

        return self.current_image >= self._scanning_job.number_of_scans or self.current_image is None

    def _get_step_to_next_scan_cycle_step(self):

        if self._scanning_effector_data.current_cycle_step.next_minor is self._scan_cycle_step:
            return SCAN_STEP.NextMajor
        else:
            return SCAN_STEP.NextMinor

    def _update_scan_cycle_step(self, step_action):

        if step_action is SCAN_STEP.NextMajor:
            self._scanning_effector_data.current_cycle_step = self._scanning_effector_data.current_cycle_step.next_major
        elif step_action is SCAN_STEP.NextMinor:
            self._scanning_effector_data.current_cycle_step = self._scanning_effector_data.current_cycle_step.next_minor

        if not step_action is SCAN_STEP.Wait:
            self._scanning_effector_data.current_step_start_time = time.time()

    def _do_wait(self):

        if self.current_image < 0:
            self._start_time = time.time()
            self._scanning_effector_data.previous_scan_time = -self._scanning_job.time_between_scans * 1.1
            self._scanning_effector_data.current_image = 0
            return SCAN_STEP.NextMajor

        project_time = time.time() - self._start_time

        if project_time >= self._scanning_job.time_between_scans:
            self._scanning_effector_data.previous_scan_time = project_time
            return  SCAN_STEP.NextMajor

        return SCAN_STEP.Wait

    def _do_wait_for_usb(self):
        if self._scanning_effector_data.usb_port:
            return  SCAN_STEP.NextMajor
        elif (time.time() - self._scanning_effector_data.current_step_start_time <
                      self._scanning_job.time_between_scans * self.WAIT_FOR_USB_TOLERANCE_FACTOR):
            return SCAN_STEP.Wait
        else:
            return SCAN_STEP.NextMinor

    def _do_wait_for_scan(self):

        if (time.time() - self._scanning_effector_data.current_step_start_time <
            self._scanning_job.time_between_scans * self.WAIT_FOR_SCAN_TOLERANCE_FACTOR)
        return SCAN_STEP.Wait

    def _do_request_scanner_on(self):

        self.pipe_effector.send("request_scanner_on", self._scanning_job.scanner)
        return SCAN_STEP.NextMinor

    def _do_request_scanner_off(self):

        self.pipe_effector.send("request_scanner_off", self._scanning_job.scanner)
        return SCAN_STEP.NextMajor

    def _do_request_first_pass_analysis(self):

        return SCAN_STEP.NextMajor

    def _do_scan(self):

        return SCAN_STEP.NextMinor

    def _setup_directory(self):

        os.makedirs(os.path.join(self._scanning_job.directory_containing_project.rstrip(os.sep),
                                 self._scanning_job.project_name))

    def _set_usb_port(self, port):

        self._scanning_effector_data.usb_port = port

    @decorators.type_lock
    def _add_scanned_image(self, index, time_stamp, path):

        self._scanning_effector_data.images_ready_for_first_pass_analysis.append((index, time_stamp, path))