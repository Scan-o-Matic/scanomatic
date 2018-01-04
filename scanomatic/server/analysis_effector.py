from __future__ import absolute_import
#
# DEPENDENCIES
#

import os
import time

#
# INTERNAL DEPENDENCIES
#

from . import proc_effector
import scanomatic.io.image_data as image_data
from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config as AppConfig
import scanomatic.image_analysis.analysis_image as analysis_image
from scanomatic.models.rpc_job_models import JOB_TYPE
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.factories.fixture_factories import (
    GrayScaleAreaModelFactory, FixturePlateFactory
)
from scanomatic.models.factories.features_factory import FeaturesFactory
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
import scanomatic.io.first_pass_results as first_pass_results
import scanomatic.io.rpc_client as rpc_client
from scanomatic.data_processing.phenotyper import remove_state_from_path

def get_label_from_analysis_model(analysis_model, id_hash):
    """Make a suitable label to show in status view

    :param analysis_model: The model
     :type analysis_model: scanomatic.models.analysis_model.AnalysisModel
    :param id_hash : The identifier of the project
     :type id_hash : str
    :return: label
    :rtype: str
    """

    root = os.path.basename(os.path.dirname(analysis_model.compilation)).replace("_", " ")
    output = analysis_model.output_directory.replace("_", " ") if analysis_model.output_directory else "analysis"
    return "{0} -> {1}, ({2})".format(root, output, id_hash[-6:])

#
# CLASSES
#


class AnalysisEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Analysis

    def __init__(self, job):
        """

        :param job: The job
         :type job : scanomatic.models.rpc_job_models.RPCjobModel
        :return:
        """
        # sys.excepthook = support.custom_traceback

        super(AnalysisEffector, self).__init__(job, logger_name="Analysis Effector")
        self._config = None
        self._job_label = get_label_from_analysis_model(job.content_model, job.id)

        self._specific_statuses['total'] = 'total'
        self._specific_statuses['current_image_index'] = 'current_image_index'

        self._allowed_calls['setup'] = self.setup

        self._redirect_logging = True
        self._reference_compilation_image_model = None

        if job.content_model:
            self._analysis_job = AnalysisModelFactory.create(**job.content_model)
        else:
            self._analysis_job = AnalysisModelFactory.create()
            self._logger.warning("No job instructions")

        self._original_model = None

        self._job.content_model = self._analysis_job

        self._scanning_instructions = None

        self._current_image_model = None
        """:type : scanomatic.models.compile_project_model.CompileImageAnalysisModel"""
        self._analysis_needs_init = True

    @property
    def current_image_index(self):
        if self._current_image_model:
            return self._current_image_model.image.index
        return -1

    @property
    def total(self):
        if self._get_is_analysing_images():
            return self._first_pass_results.total_number_of_images
        return -1

    def _get_is_analysing_images(self):
        return self._allow_start and hasattr(self, "_first_pass_results") and self._first_pass_results

    @property
    def progress(self):

        total = float(self.total)
        initiation_weight = 1

        if total > 0 and self._current_image_model:
            return (total - self.current_image_index + initiation_weight) / float(total + initiation_weight)

        return 0.0

    def next(self):

        if self.waiting:
            return super(AnalysisEffector, self).next()
        elif not self._stopping:
            if self._analysis_needs_init:
                return self._setup_first_iteration()
            elif not self._stopping:
                if not self._analyze_image():
                    self._stopping = True
                return not self._stopping
            else:
                self._finalize_analysis()
                raise StopIteration
        else:
            self._finalize_analysis()
            raise StopIteration

    def _finalize_analysis(self):

        self._logger.info("ANALYSIS, Full analysis took {0} minutes".format(
            ((time.time() - self._start_time) / 60.0)))

        self._logger.info('Analysis completed at ' + str(time.time()))

        if self._analysis_job.chain:

            try:
                rc = rpc_client.get_client(admin=True)
                if rc.create_feature_extract_job(FeaturesFactory.to_dict(FeaturesFactory.create(
                        analysis_directory=self._analysis_job.output_directory,
                        email=self._analysis_job.email))):

                    self._logger.info("Enqueued feature extraction job")
                else:
                    self._logger.warning("Enqueing of feature extraction job refused")
            except:
                self._logger.error("Could not spawn analysis at directory {0}".format(
                    self._analysis_job.output_directory))
        else:
            self._mail("Scan-o-Matic: Analysis for project '{project_name}' done.",
                       """This is an automated email, please don't reply!

The project '{compile_instructions}' on """ + AppConfig().computer_human_name +
                       """ is done and no further action requested.

All the best,

Scan-o-Matic""", self._analysis_job)

        self._running = False

    def _analyze_image(self):

        scan_start_time = time.time()
        image_model = self._first_pass_results.get_next_image_model()

        if image_model is None:
            self._stopping = True
            return False
        elif self._reference_compilation_image_model is None:
            # Using the first recieved model / last in project as reference model.
            # Used for one_time type of analysis settings
            self._reference_compilation_image_model = image_model

        # TODO: Verify that this isn't the thing causing the capping!
        if (image_model.fixture.grayscale is None or
                image_model.fixture.grayscale.values is None):

            self._logger.error(
                "No grayscale analysis results for '{0}' means image not included in analysis".format(
                image_model.image.path))
            return True


        # Overwrite grayscale with previous if has been requested
        if self._analysis_job.one_time_grayscale:

            self._logger.info("Using the grayscale detected on {0} for {1}".format(
                self._reference_compilation_image_model.image.path,
                image_model.image.path))

            image_model.fixture.grayscale = GrayScaleAreaModelFactory.copy(
                    self._reference_compilation_image_model.fixture.grayscale)

        # Overwrite plate positions if requested
        if self._analysis_job.one_time_positioning:

            self._logger.info("Using plate positions detected on {0} for {1}".format(
                self._reference_compilation_image_model.image.path,
                image_model.image.path))

            image_model.fixture.orientation_marks_x = \
                [v for v in self._reference_compilation_image_model.fixture.orientation_marks_x]
            image_model.fixture.orientation_marks_y = \
                [v for v in self._reference_compilation_image_model.fixture.orientation_marks_y]
            image_model.fixture.plates = \
                [FixturePlateFactory.copy(m) for m in self._reference_compilation_image_model.fixture.plates]

        first_image_analysed = self._current_image_model is None
        self._current_image_model = image_model

        self._logger.info("ANALYSIS, Running analysis on '{0}'".format(image_model.image.path))

        self._image.analyse(image_model)
        self._logger.info("Analysis took {0}, will now write out results.".format(time.time() - scan_start_time))

        features = self._image.features

        if features is None:
            self._logger.warning("Analysis features not set up correctly")

        image_data.ImageData.write_times(self._analysis_job, image_model, overwrite=first_image_analysed)
        if not image_data.ImageData.write_image(self._analysis_job, image_model, features):
            self._stopping = True
            self._logger.critical("Terminating analysis since output can't be stored")
            return False

        self._logger.info("Image took {0} seconds".format(time.time() - scan_start_time))

        return True

    def _setup_first_iteration(self):

        self._start_time = time.time()

        self._first_pass_results = first_pass_results.CompilationResults(
            self._analysis_job.compilation,
            self._analysis_job.compile_instructions)

        try:
            os.makedirs(self._analysis_job.output_directory)
        except OSError, e:
            if e.errno == os.errno.EEXIST:
                self._logger.warning(
                    "Output directory exists, previous data will be wiped")
            else:
                self._running = False
                self._logger.critical(
                    "Can't create output directory '{0}'".format(
                        self._analysis_job.output_directory))
                raise StopIteration

        if self._redirect_logging:
            self._logger.info(
                "{0} is setting up, output will be directed to {1}".format(
                    self._analysis_job, Paths().analysis_run_log))

            log_path = os.path.join(
                self._analysis_job.output_directory, Paths().analysis_run_log)
            self._logger.set_output_target(
                log_path, catch_stdout=True, catch_stderr=True, buffering=0)
            self._logger.surpress_prints = False
            self._log_file_path = log_path

        if (len(self._first_pass_results.plates) !=
                len(self._analysis_job.pinning_matrices)):
            self._filter_pinning_on_included_plates()

        AnalysisModelFactory.serializer.dump(
            self._original_model, os.path.join(
                self._analysis_job.output_directory,
                Paths().analysis_model_file))

        self._logger.info("Will remove previous files")

        self._remove_files_from_previous_analysis()

        self._image = analysis_image.ProjectImage(
            self._analysis_job, self._first_pass_results)

        # TODO: Need rework to handle gridding of diff times for diff plates

        if not self._image.set_grid():
            self._stopping = True

        self._analysis_needs_init = False

        self._logger.info(
            'Primary data format will save {0}:{1}'.format(
                self._analysis_job.image_data_output_item,
                self._analysis_job.image_data_output_measure))

        return True

    def _filter_pinning_on_included_plates(self):

        included_indices = tuple(p.index for p in self._first_pass_results.plates)
        self._analysis_job.pinning_matrices = [pm for i, pm in enumerate(self._analysis_job.pinning_matrices)
                                               if i in included_indices]
        self._logger.warning("Inconsistency in number of plates reported in analysis instruction and compilation." +
                             " Asuming pinning to be {0}".format(self._analysis_job.pinning_matrices))

        self._original_model.pinning_matrices = self._analysis_job.pinning_matrices

    def _remove_files_from_previous_analysis(self):

        n = 0
        for p in image_data.ImageData.iter_image_paths(self._analysis_job.output_directory):
            os.remove(p)
            n += 1

        if n:
            self._logger.info("Removed {0} pre-existing image data files".format(n))

        times_path = os.path.join(self._analysis_job.output_directory, Paths().image_analysis_time_series)
        try:
            os.remove(times_path)
        except (IOError, OSError):
            pass
        else:
            self._logger.info("Removed pre-existing time data file")

        for i, _ in enumerate(self._analysis_job.pinning_matrices):

            for filename_pattern in (Paths().grid_pattern, Paths().grid_size_pattern,
                                     Paths().experiment_grid_error_image,
                                     Paths().experiment_grid_image_pattern):

                grid_path = os.path.join(self._analysis_job.output_directory, filename_pattern).format(i + 1)
                try:
                    os.remove(grid_path)
                except (IOError, OSError):
                    pass
                else:
                    self._logger.info("Removed pre-existing grid file '{0}'".format(grid_path))

        remove_state_from_path(self._analysis_job.output_directory)

    def setup(self, job, redirect_logging=True):

        if self._running:
            self.add_message("Cannot change settings while running")
            return

        self._redirect_logging = redirect_logging

        if not self._analysis_job.output_directory:
            AnalysisModelFactory.set_default(self._analysis_job, [self._analysis_job.FIELD_TYPES.output_directory])
            self._logger.info("Using default '{0}' output directory".format(self._analysis_job.output_directory))
        if not self._analysis_job.compile_instructions:
            self._analysis_job.compile_instructions = \
                Paths().get_project_compile_instructions_path_from_compilation_path(self._analysis_job.compilation)
            self._logger.info("Setting to default compile instructions path {0}".format(
                self._analysis_job.compile_instructions))

        allow_start = AnalysisModelFactory.validate(self._analysis_job)

        self._original_model = AnalysisModelFactory.copy(self._analysis_job)
        AnalysisModelFactory.set_absolute_paths(self._analysis_job)

        try:
            self._scanning_instructions = ScanningModelFactory.serializer.load_first(
                Paths().get_scan_instructions_path_from_compile_instructions_path(
                    self._analysis_job.compile_instructions))
        except IOError:
            self._logger.warning("No information found about how the scanning was done," +
                                 " using empty instructions instead")

        if not self._scanning_instructions:
            self._scanning_instructions = ScanningModelFactory.create()

        self.ensure_default_values_if_missing()

        self._allow_start = allow_start
        if not self._allow_start:
            self._logger.error("Can't perform analysis; instructions don't validate.")
            for bad_instruction in AnalysisModelFactory.get_invalid(self._analysis_job):
                self._logger.error("Bad value {0}={1}".format(bad_instruction, self._analysis_job[bad_instruction.name]
                                                              ))
            self.add_message("Can't perform analysis; instructions don't validate.")
            self._stopping = True

    def ensure_default_values_if_missing(self):

        if not self._analysis_job.image_data_output_measure:
            AnalysisModelFactory.set_default(
                self._analysis_job,
                [self._analysis_job.FIELD_TYPES.image_data_output_measure])
        if not self._analysis_job.image_data_output_item:
            AnalysisModelFactory.set_default(
                self._analysis_job,
                [self._analysis_job.FIELD_TYPES.image_data_output_item])
