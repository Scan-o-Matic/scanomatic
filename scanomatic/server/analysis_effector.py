#
# DEPENDENCIES
#

import os
import time

#
# INTERNAL DEPENDENCIES
#

import proc_effector
import scanomatic.io.xml.writer as xml_writer
import scanomatic.io.image_data as image_data
from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config as AppConfig
import scanomatic.imageAnalysis.support as support
import scanomatic.imageAnalysis.analysis_image as analysis_image
from scanomatic.models.rpc_job_models import JOB_TYPE
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory, XMLModelFactory
from scanomatic.models.factories.fixture_factories import GrayScaleAreaModelFactory, FixturePlateFactory
from scanomatic.models.factories.features_factory import FeaturesFactory
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
import scanomatic.io.first_pass_results as first_pass_results
import scanomatic.io.rpc_client as rpc_client

#
# CLASSES
#


class AnalysisEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Analysis

    def __init__(self, job):

        # sys.excepthook = support.custom_traceback

        super(AnalysisEffector, self).__init__(job, logger_name="Analysis Effector")
        self._config = None
        self._job_label = job.content_model.compilation

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

        self._job.content_model = self._analysis_job

        self._scanning_instructions = None

        self._focus_graph = None
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

        # TODO: Verify this is correct, may underestimate progress
        if total > 0 and self._current_image_model:
            return (total - self.current_image_index) / float(total + initiation_weight)

        return 0.0

    def next(self):

        if self.waiting:
            return super(AnalysisEffector, self).next()
        elif not self._stopping:
            if self._analysis_needs_init:
                return self._setup_first_iteration()
            elif not self._stopping:
                return self._analyze_image()
            else:
                return self._finalize_analysis()
        else:
            return self._finalize_analysis()

    def _finalize_analysis(self):

            self._xmlWriter.close()

            if self._focus_graph is not None:
                self._focus_graph.finalize()

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
            raise StopIteration

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
        if image_model.fixture.grayscale is None or image_model.fixture.grayscale.values is None:
            self._logger.error("No grayscale analysis results for '{0}' means image not included in analysis".format(
                image_model.image.path))
            return True

        image_model.fixture.grayscale.values = image_model.fixture.grayscale.values[::-1]

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

        features = self._image.features

        if features is None:
            self._logger.warning("Analysis features not set up correctly")

        image_data.ImageData.write_times(self._analysis_job, image_model, overwrite=first_image_analysed)
        if not image_data.ImageData.write_image(self._analysis_job, image_model, features):
            self._stopping = True
            self._logger.critical("Terminating analysis since output can't be stored")
            return False

        self._xmlWriter.write_image_features(image_model, features)

        if self._focus_graph:

            self._focus_graph.add_image(self._image.watch_source, self._image.watch_blob)

        self._logger.info("Image took {0} seconds".format(time.time() - scan_start_time))

        return True

    def _setup_first_iteration(self):

        self._start_time = time.time()

        self._first_pass_results = first_pass_results.CompilationResults(
            self._analysis_job.compilation, self._analysis_job.compile_instructions)

        try:
            os.makedirs(self._analysis_job.output_directory)
        except OSError, e:
            if e.errno == os.errno.EEXIST:
                self._logger.warning("Output directory exists, previous data will be wiped")
            else:
                self._running = False
                self._logger.critical("Can't create output directory '{0}'".format(self._analysis_job.output_directory))
                raise StopIteration

        if self._redirect_logging:
            self._logger.info("{0} is setting up, output will be directed to {1}".format(self._analysis_job,
                                                                                         Paths().analysis_run_log))
            self._logger.set_output_target(
                os.path.join(self._analysis_job.output_directory, Paths().analysis_run_log),
                catch_stdout=True, catch_stderr=True)

            self._logger.surpress_prints = True

        self._logger.info("Will remove previous files")

        self._remove_files_from_previous_analysis()

        if self._analysis_job.focus_position is not None:
            self._focus_graph = support.Watch_Graph(
                self._analysis_job.focus_position, self._analysis_job.output_directory)

        self._xmlWriter = xml_writer.XML_Writer(
            self._analysis_job.output_directory, self._analysis_job.xml_model)

        if self._xmlWriter.get_initialized() is False:

            self._logger.critical('XML writer failed to initialize')
            self._xmlWriter.close()
            self._running = False

            raise StopIteration

        self._image = analysis_image.ProjectImage(self._analysis_job, self._first_pass_results.compile_instructions)

        self._xmlWriter.write_header(self._scanning_instructions, self._first_pass_results.plates)
        self._xmlWriter.write_segment_start_scans()

        index_for_gridding = self._get_index_for_gridding()

        if not self._image.set_grid(self._first_pass_results[index_for_gridding]):
            self._stopping = True

        self._analysis_needs_init = False

        self._logger.info('Primary data format will save {0}:{1}'.format(self._analysis_job.image_data_output_item,
                                                                         self._analysis_job.image_data_output_measure))

        self._logger.info('Analysis saved in XML-slimmed will be {0}:{1}'.format(
            self._analysis_job.xml_model.slim_compartment, self._analysis_job.xml_model.slim_measure))

        self._logger.info('Compartments excluded from big XML are {0}'.format(
            self._analysis_job.xml_model.exclude_compartments))

        self._logger.info('Measures excluded from big XML are {0}'.format(
            self._analysis_job.xml_model.exclude_measures))

        return True

    def _remove_files_from_previous_analysis(self):

        for p in image_data.ImageData.iter_image_paths(self._analysis_job.output_directory):
            os.remove(p)
            self._logger.info("Removed pre-existing file '{0}'".format(p))

    def _get_index_for_gridding(self):

        if self._analysis_job.grid_images:
            pos = max(self._analysis_job.grid_images)
            if pos >= len(self._first_pass_results):
                pos = self._first_pass_results.last_index
        else:

            pos = self._first_pass_results.last_index

        return pos

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

        if not self._analysis_job.xml_model.slim_measure:
            XMLModelFactory.set_default(self._analysis_job.xml_model,
                                        [self._analysis_job.xml_model.FIELD_TYPES.slim_measure])
        if not self._analysis_job.xml_model.slim_compartment:
            XMLModelFactory.set_default(self._analysis_job.xml_model,
                                        [self._analysis_job.xml_model.FIELD_TYPES.slim_compartment])
        if not self._analysis_job.image_data_output_measure:
            AnalysisModelFactory.set_default(self._analysis_job,
                                             [self._analysis_job.FIELD_TYPES.image_data_output_measure])
        if not self._analysis_job.image_data_output_item:
            AnalysisModelFactory.set_default(self._analysis_job,
                                             [self._analysis_job.FIELD_TYPES.image_data_output_item])