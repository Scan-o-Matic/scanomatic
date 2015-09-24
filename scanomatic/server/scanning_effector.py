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
from scanomatic.models.scanning_model import SCAN_CYCLE, SCAN_STEP, ScanningModelEffectorData
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
from scanomatic.models.compile_project_model import COMPILE_ACTION
from scanomatic.io import scanner_manager
from scanomatic.io import sane
from scanomatic.io import paths
from threading import Thread
import scanomatic.io.rpc_client as rpc_client
from scanomatic.models.factories import compile_project_factory
from scanomatic.io import mail
from scanomatic.io.app_config import Config as AppConfig

JOBS_CALL_SET_USB = "set_usb"
SECONDS_PER_MINUTE = 60.0

FILE_SIZE_DEVIATION_ALLOWANCE = 0.2
TOO_SMALL_SIZE = 1024 * 1024
DISKSPACE_MARGIN_FACTOR = 5



class ScannerEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Scan
    WAIT_FOR_USB_TOLERANCE_FACTOR = 0.33
    WAIT_FOR_SCAN_TOLERANCE_FACTOR = 0.5
    WAIT_FOR_NEXT_SCAN = 1.0

    def __init__(self, job):

        # sys.excepthook = support.custom_traceback

        super(ScannerEffector, self).__init__(job, logger_name="Scanner Effector")

        self._specific_statuses['total'] = 'total_images'
        self._specific_statuses['currentImage'] = 'current_image'

        self._allowed_calls['setup'] = self.setup
        self._allowed_calls[JOBS_CALL_SET_USB] = self._set_usb_port

        self._scanning_job = job.content_model
        """:type : scanomatic.models.scanning_model.ScanningModel"""
        self._job_label = "'{0}' on scanner {1}".format(self._scanning_job.project_name, self._scanning_job.scanner)
        self._scanning_effector_data = ScanningModelEffectorData()
        self._rpc_client = rpc_client.get_client(admin=True)
        self._scanner = None
        self._scan_cycle_step = SCAN_CYCLE.Wait

        self._scan_cycle = {
            SCAN_CYCLE.Wait: self._do_wait,
            SCAN_CYCLE.RequestScanner: self._do_request_scanner_on,
            SCAN_CYCLE.RequestScannerOff: self._do_request_scanner_off,
            SCAN_CYCLE.RequestProjectCompilation: self._do_request_project_compilation,
            SCAN_CYCLE.Scan: self._do_scan,
            SCAN_CYCLE.ReportNotObtainedUSB: self._do_report_error_obtaining_scanner,
            SCAN_CYCLE.ReportScanError: self._do_report_error_scanning,
            SCAN_CYCLE.WaitForScanComplete: self._do_wait_for_scan,
            SCAN_CYCLE.WaitForUSB: self._do_wait_for_usb,
            SCAN_CYCLE.VerifyImageSize: self._do_verify_image_size,
            SCAN_CYCLE.VerifyDiskspace: self._do_verify_image_size
        }

    def setup(self, job, redirect_logging=True):

        job = RPC_Job_Model_Factory.serializer.load_serialized_object(job)[0]
        paths_object = paths.Paths()
        self._scanning_job.id = job.id
        self._setup_directory()

        if redirect_logging:
            self._logger.info("{0} is setting up; logging will be directed to file".format(job))
            self._logger.set_output_target(
                os.path.join(self._project_directory,
                             paths_object.scan_log_file_pattern.format(self._scanning_job.project_name)),
                catch_stdout=True, catch_stderr=True)

            self._logger.surpress_prints = True

        self._logger.info("Doing setup")

        self._scanning_effector_data.current_image_path_pattern = os.path.join(
            self._project_directory,
            paths_object.experiment_scan_image_pattern)

        self._scanner = sane.SaneBase(scan_mode=self._scanning_job.mode, model=self._scanning_job.scanner_hardware)

        self._scanning_effector_data.compile_project_model = compile_project_factory.CompileProjectFactory.create(
            compile_action=COMPILE_ACTION.Initiate,
            images=self._scanning_effector_data.images_ready_for_first_pass_analysis,
            path=paths_object.get_project_settings_path_from_scan_model(self._scanning_job))

        scan_project_file_path = os.path.join(
            self._project_directory,
            paths_object.scan_project_file_pattern.format(self._scanning_job.project_name))

        if ScanningModelFactory.serializer.dump(self._scanning_job, scan_project_file_path):

            self._logger.info("Saved project settings to '{0}'".format(scan_project_file_path))

        else:

            self._logger.error("Could not save project settings to '{0}'".format(scan_project_file_path))
        self._allow_start = True

    @property
    def progress(self):

        global SECONDS_PER_MINUTE
        run_time = self.run_time
        if run_time <= 0 or not self._allow_start:
            return 0
        else:

            # Actual duration is expected to be one less than the number of scans plus duration of first and last scan
            # so 0.5 is a rough estimate

            return run_time / ((self._scanning_job.number_of_scans - 0.5)
                               * self._scanning_job.time_between_scans * SECONDS_PER_MINUTE)

    @property
    def total_images(self):

        return self._scanning_job.number_of_scans

    @property
    def current_image(self):

        return self._scanning_effector_data.current_image

    def next(self):

        if self.waiting:
            return super(ScannerEffector, self).next()
        elif not self._stopping:
            try:
                step_action = self._scan_cycle[self._scanning_effector_data.current_cycle_step]()
            except KeyError:
                self._logger.warning("Error performing step {0}, no known method for that step".format(
                    self._scanning_effector_data.current_cycle_step))

                step_action = self._get_step_to_next_scan_cycle_step()

            self._update_scan_cycle_step(step_action)
        else:
            self._logger.info("Interrupted progress {0} at {1} ({2})".format(
                self._scanning_job,
                self._scanning_effector_data.current_cycle_step,
                self._scanning_effector_data.previous_scan_cycle_start))

            self._scanning_effector_data.compile_project_model.compile_action = COMPILE_ACTION.AppendAndSpawnAnalysis

            self._do_request_project_compilation()

            self._scanning_effector_data.current_image = None

        if self._job_completed:
            self._stopping = True

            if self._scanning_effector_data.images_ready_for_first_pass_analysis:

                self._scanning_effector_data.compile_project_model.compile_action = COMPILE_ACTION.AppendAndSpawnAnalysis
                self._do_request_project_compilation()

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
        elif step_action is SCAN_STEP.TruncateIteration:
            self._scanning_effector_data.current_cycle_step = SCAN_CYCLE.Wait

        if step_action is None:
            self._logger.error("Scan step {0} failed to return a valid step action".format((
                self._scanning_effector_data.current_cycle_step)))
        elif step_action is not SCAN_STEP.Wait:
            self._scanning_effector_data.current_step_start_time = time.time()

    def _do_wait(self):

        if self.current_image < 0:
            self._start_time = time.time()
            self._scanning_effector_data.previous_scan_cycle_start = self.run_time
            self._scanning_effector_data.current_image = 0
            self._logger.info("Making initial scan")
            return SCAN_STEP.NextMajor

        elif not self._should_continue_waiting(self.WAIT_FOR_NEXT_SCAN, delta_time=self.time_since_last_scan):
            self._logger.info("Scan cycle {0} initiated {1}s after previous scan (sought interval {2} min)".format(
                self.current_image,
                self.time_since_last_scan,
                self._scanning_job.time_between_scans
            ))
            self._scanning_effector_data.previous_scan_cycle_start = self.run_time

            return SCAN_STEP.NextMajor
        else:
            return SCAN_STEP.Wait

    def _do_wait_for_usb(self):

        if self._scanning_effector_data.usb_port:
            self._logger.info("Job {0} knows its USB".format(self._scanning_job.id))
            if self._scanning_effector_data.warned_scanner_usb:
                self._scanning_effector_data.warned_scanner_usb = False
                self._mail("Scan-o-Matic: Resolved project '{project_name}' could not acquire its Scanner",
                           """This is an automated email, please don't reply!

The project '{project_name}' now manages to power up scanner {scanner} again.

All the best,

Scan-o-Matic""")
            return SCAN_STEP.NextMajor
        elif self._should_continue_waiting(self.WAIT_FOR_USB_TOLERANCE_FACTOR):
            return SCAN_STEP.Wait
        else:
            self._logger.info("Job {0} gave up waiting usb after {1:.2f} min".format(
                self._scanning_job.id, self.scan_cycle_step_duration / 60.0))
            return SCAN_STEP.NextMinor

    def _do_wait_for_scan(self):

        if self._scan_completed:
            if self._scanning_effector_data.scan_success:
                self._logger.info("Completed scanning image {0} located {1}".format(
                    self.current_image, self._scanning_effector_data.current_image_path))
                self._add_scanned_image(self.current_image, self._scanning_effector_data.current_scan_time,
                                        self._scanning_effector_data.current_image_path)

                if self._scanning_effector_data.warned_scanner_error:
                    self._scanning_effector_data.warned_scanner_error = False
                    self._mail("Scan-o-Matic: Resolved project '{project_name}' error while scanning",
                               """This is an automated email, please don't reply!

The project '{project_name}' now managed to successfully scan an image again.

All the best,

Scan-o-Matic""")
                return SCAN_STEP.NextMajor
            else:
                return SCAN_STEP.NextMinor

        elif self._should_continue_waiting(self.WAIT_FOR_SCAN_TOLERANCE_FACTOR):
            return SCAN_STEP.Wait
        else:
            return SCAN_STEP.NextMinor

    @property
    def _scan_completed(self):

        return not self._scanning_effector_data.scanning_thread.is_alive()

    def _do_report_error_scanning(self):

        self._logger.info("Job {0} reports scanning error".format(self._scanning_job.id))
        self._logger.error("Could not scan file {0}".format(self._scanning_effector_data.current_image_path))

        if not self._scanning_effector_data.warned_scanner_error:
            self._scanning_effector_data.warned_scanner_error = True
            self._mail("Scan-o-Matic: Project '{project_name}' error while scanning",
                       """This is an automated email, please don't reply!

The project '{project_name}' reports an error while scanning.
Please hurry to correct this so that the project won't be spoiled.

The scanning project will attempt a new scan in {time_between_scans} minutes,
but note that you won't be warned again if the error persists.

Instead you will be notified when/if error is resolved.

All the best,

Scan-o-Matic""")

        return SCAN_STEP.NextMajor

    def _do_report_error_obtaining_scanner(self):

        self._logger.error("Server never gave me my scanner.")
        self._do_request_scanner_off()

        if not self._scanning_effector_data.warned_scanner_usb:
            self._scanning_effector_data.warned_scanner_usb = True
            self._mail("Scan-o-Matic: Project '{project_name}' could not acquire its Scanner",
                       """This is an automated email, please don't reply!

The project '{project_name}' could not get scanner {scanner} powered up.
Please hurry to correct this so that the project won't be spoiled.

The scanning project will attempt a new scan in {time_between_scans} minutes,
but note that you won't be warned again if the error persists.

Instead you will be notified when/if error is resolved.

All the best,

Scan-o-Matic""")

        return SCAN_STEP.TruncateIteration

    def _should_continue_waiting(self, max_between_scan_fraction, delta_time=None):

        global SECONDS_PER_MINUTE

        if delta_time is None:
            delta_time = self.scan_cycle_step_duration

        return (delta_time <
                self._scanning_job.time_between_scans * SECONDS_PER_MINUTE * max_between_scan_fraction)

    @property
    def time_since_last_scan(self):

        return self.run_time - self._scanning_effector_data.previous_scan_cycle_start

    @property
    def scan_cycle_step_duration(self):

        return time.time() - self._scanning_effector_data.current_step_start_time

    def _do_verify_image_size(self):

        def get_size_of_last_image():
            try:
                return os.stat(self._scanning_effector_data.current_image_path).st_size
            except OSError:
                return 0

        current_size = get_size_of_last_image()
        largest_known_size = max(current_size, self._scanning_effector_data.known_file_size)

        if current_size < TOO_SMALL_SIZE:

            if self._scanning_effector_data.warned_file_size is False:
                self._scanning_effector_data.warned_file_size = True
                self._mail("Scan-o-Matic: Project '{project_name}' got suspicious image",
                           """This is an automated email, please don't reply!

The project '{project_name}' got an image of very small size.

""" + "{0}:\t{1} bytes\n".format(self._scanning_effector_data.current_image_path, current_size) + """

Several reasons are probable:

   1) The hard drive is full
   2) The scanner lost power or crashed while acquiring the image

All the best,

Scan-o-Matic""")

        elif (self._scanning_effector_data.known_file_size and
                abs(self._scanning_effector_data.known_file_size - current_size) / largest_known_size >
                FILE_SIZE_DEVIATION_ALLOWANCE):

            if self._scanning_effector_data.warned_file_size is False:
                self._scanning_effector_data.warned_file_size = True
                self._mail("Scan-o-Matic: Project '{project_name}' got suspicious image",
                           """This is an automated email, please don't reply!

The project '{project_name}' got an image of unexpected size.

""" + "{0}:\t{1} bytes\n\n".format(self._scanning_effector_data.current_image_path, current_size) +
                           "Previously, the largest size was {0}, such deviations aren't expected.".format(
                               self._scanning_effector_data.known_file_size) + """
Several reasons are probable:

   1) The hard drive is full
   2) The scanner lost power or crashed while acquiring the image

All the best,

Scan-o-Matic""")

        elif self._scanning_effector_data.warned_file_size is True:

            self._scanning_effector_data.warned_file_size = False
            self._mail("Scan-o-Matic: Resolved, project '{project_name}' now got normal image",
                       """This is an automated email, please don't reply!

The project '{project_name}' got image of expected size again! Yay.

All the best,

Scan-o-Matic""")

        self._scanning_effector_data.known_file_size = largest_known_size
        return SCAN_STEP.NextMinor

    def _do_verify_discspace(self):

        def get_free_space():

            try:
                vfs = os.statvfs(self._scanning_job.directory_containing_project)
                return vfs.f_frsize * vfs.f_bavail
            except OSError:
                return 0

        if self._scanning_effector_data.known_file_size and not self._scanning_effector_data.warned_discspace:

            bytes_needed = (self._scanning_job.number_of_scans - self._scanning_effector_data.current_image) * \
                self._scanning_effector_data.known_file_size

            if bytes_needed * DISKSPACE_MARGIN_FACTOR > get_free_space():
                self._scanning_effector_data.warned_discspace = True
                self._mail("Scan-o-Matic: Project '{project_name}' may not have enough disc space",
                           """This is an automated email, please don't reply!

The project '{project_name}' is reporting that the remaining space the
drive it is saving its data to may not be enough for the remainder of the project
to complete.

Note that this is an estimate, as the project is unaware of other projects running
at the same time, but please verify remaining space.

""" + "Report triggered after acquiring image index {0}".format(
                               self._scanning_effector_data.current_image) + """ of {number_of_scans}.

No further warnings about disc space will be sent for this project.

All the best,

Scan-o-Matic""")

        return SCAN_STEP.NextMajor

    def _do_request_scanner_on(self):

        self._logger.info("Job {0} requested scanner on".format(self._scanning_job.id))
        self.pipe_effector.send(scanner_manager.JOB_CALL_SCANNER_REQUEST_ON, self._scanning_job.id)
        return SCAN_STEP.NextMinor

    def _do_request_scanner_off(self):

        self._logger.info("Job {0} requested scanner off".format(self._scanning_job.id))
        self.pipe_effector.send(scanner_manager.JOB_CALL_SCANNER_REQUEST_OFF, self._scanning_job.id)
        self._scanning_effector_data.usb_port = ""
        self._scanning_effector_data.current_image += 1
        return SCAN_STEP.NextMajor

    def _do_request_project_compilation(self):
        """Requests compile project if there was a fixture given.

                If it is the first request of compilation, the COMPILE_ACTION is set to initiate from
                the setup-method.
        """
        if self._scanning_job.fixture:

            compile_job_id = self._rpc_client.create_compile_project_job(
                compile_project_factory.CompileProjectFactory.to_dict(
                    self._scanning_effector_data.compile_project_model))

            if compile_job_id:

                # Images start at 0, next to last has index total - 2
                next_image_is_last = self._scanning_job.number_of_scans - 2 == \
                    self._scanning_effector_data.current_image

                if next_image_is_last:
                    self._scanning_effector_data.compile_project_model.compile_action = \
                        COMPILE_ACTION.AppendAndSpawnAnalysis
                else:
                    self._scanning_effector_data.compile_project_model.compile_action = COMPILE_ACTION.Append
                self._scanning_effector_data.compile_project_model.start_condition = compile_job_id
                self._scanning_effector_data.images_ready_for_first_pass_analysis.clear()
                self._logger.info("Job {0} created compile project job".format(self._scanning_job.id))
            else:
                self._logger.warning("Failed to create a compile project job, refused by server")
        else:
            self._logger.info("Not enqueing any project compilation since no fixture used")
        return SCAN_STEP.NextMajor

    def _do_scan(self):

        if self._scanning_effector_data.usb_port:

            self._scanning_effector_data.current_scan_time = self.run_time
            self._scanning_effector_data.current_image_path = \
                self._scanning_effector_data.current_image_path_pattern.format(
                    self._scanning_job.project_name, str(self._scanning_effector_data.current_image).zfill(4),
                    self._scanning_effector_data.current_scan_time)

            self._scanning_effector_data.scanning_thread = Thread(target=self._scan_thread)
            self._scanning_effector_data.scanning_thread.start()
            self._logger.info("Job {0} started scan".format(self._scanning_job.id))
            return SCAN_STEP.NextMinor
        else:
            self._logger.error("No registered USB port when attempting to scan")
            return SCAN_STEP.NextMajor

    def _scan_thread(self):

        self._scanning_effector_data.scan_success = self._scanner.AcquireByFile(
            scanner=self._scanning_effector_data.usb_port,
            filename=self._scanning_effector_data.current_image_path)

    def _setup_directory(self):

        os.makedirs(self._project_directory)

    @property
    def _project_directory(self):

        return os.path.join(self._scanning_job.directory_containing_project.rstrip(os.sep),
                            self._scanning_job.project_name)

    def _set_usb_port(self, port):

        self._logger.info("Got an usb port '{0}'".format(port))
        self._scanning_effector_data.usb_port = port

    def _add_scanned_image(self, index, time_stamp, path):

        image_model = compile_project_factory.CompileImageFactory.create(
            index=index, time_stamp=time_stamp, path=path)

        self._scanning_effector_data.images_ready_for_first_pass_analysis.append(image_model)

    def _mail(self, title_template, message_template):

        def _do_mail(title_template, message_template, scanning_job_model):

            if not scanning_job_model.email:
                return

            if AppConfig().mail_server:
                server = mail.get_server(AppConfig().mail_server, smtp_port=AppConfig().mail_port,
                                         login=AppConfig().mail_user, password=AppConfig().mail_password)
            else:
                server = None

            mail.mail(scanning_job_model.email if AppConfig().mail_user is None else AppConfig().mail_user,
                      scanning_job_model.email,
                      title_template.format(**scanning_job_model),
                      message_template.format(**scanning_job_model),
                      server=server)

        Thread(target=_do_mail, args=(title_template, message_template, self._scanning_job)).start()