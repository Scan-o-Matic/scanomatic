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
import time

#
# INTERNAL DEPENDENCIES
#

import proc_effector
import scanomatic.io.xml.writer as xml_writer
import scanomatic.io.image_data as image_data
from scanomatic.io.paths import Paths
import scanomatic.imageAnalysis.support as support
import scanomatic.imageAnalysis.analysis_image as analysis_image
from scanomatic.models.rpc_job_models import JOB_TYPE
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
import scanomatic.io.first_pass_results as first_pass_results


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

        if job.content_model:
            self._analysis_job = AnalysisModelFactory.create(**job.content_model)
        else:
            self._analysis_job = AnalysisModelFactory.create()
            self._logger.warning("No job instructions")

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

    @property
    def waiting(self):
        return not(self._allow_start and self._running)

    def next(self):
        if self.waiting:
            return super(AnalysisEffector, self).next()
        elif self._analysis_needs_init:
            return self._setup_first_iteration()
        elif not self._stopping:
            return self._analyze_image()
        else:
            return self._finalize_analysis()

    def _finalize_analysis(self):

            self._xmlWriter.close()

            if self._focus_graph is not None:
                self._focus_graph.finalize()

            self._logger.info("ANALYSIS, Full analysis took {0} minutes".format(
                ((time.time() - self._startTime) / 60.0)))

            self._logger.info('Analysis completed at ' + str(time.time()))

            self._running = False
            raise StopIteration

    def _analyze_image(self):

        scan_start_time = time.time()
        image_model = self._first_pass_results.get_next_image_model()

        # TODO: Hack to patch flipped calibration X axis
        image_model.fixture.grayscale.values = image_model.fixture.grayscale.values[::-1]

        first_image_analysed = self._current_image_model is None
        self._current_image_model = image_model
        if not image_model:
            self._stopping = True
            return True

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

        self._startTime = time.time()

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

        self._logger.info("{0} is setting up, output will be directed to {1}".format(self._analysis_job,
                                                                                     Paths().analysis_run_log))
        self._logger.set_output_target(
            os.path.join(self._analysis_job.output_directory, Paths().analysis_run_log),
            catch_stdout=True, catch_stderr=True)

        self._remove_files_from_previous_analysis()

        self._logger.surpress_prints = True

        if self._analysis_job.focus_position is not None:
            self._focus_graph = support.Watch_Graph(
                self._analysis_job.focus_position, self._analysis_job.output_directory)

        self._xmlWriter = xml_writer.XML_Writer(
            self._analysis_job.output_directory, self._analysis_job.xml_model)

        if self._xmlWriter.get_initialized() is False:

            self._logger.critical('ANALYSIS: XML writer failed to initialize')
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

    def setup(self, job):

        if self._running:
            self.add_message("Cannot change settings while running")
            return

        job = RPC_Job_Model_Factory.serializer.load_serialized_object(job)[0]

        if not self._analysis_job.compile_instructions:
            self._analysis_job.compile_instructions = \
                Paths().get_project_compile_instructions_path_from_compilation_path(self._analysis_job.compilation)
            self._logger.info("Setting to default compile instructions path {0}".format(
                self._analysis_job.compile_instructions))

        allow_start = AnalysisModelFactory.validate(self._analysis_job)

        AnalysisModelFactory.set_absolute_paths(self._analysis_job)

        try:
            self._scanning_instructions = ScanningModelFactory.serializer.load(
                Paths().get_scan_instructions_path_from_compile_instructions_path(
                    self._analysis_job.compile_instructions))[0]
        except IndexError:
            self._logger.warning("No information found about how the scanning was done")

        self._allow_start = allow_start
        if not self._allow_start:
            self._logger.error("Can't perform analysis; instructions don't validate.")
            for bad_instruction in AnalysisModelFactory.get_invalid(self._analysis_job):
                self._logger.error("Bad value {0}={1}".format(bad_instruction, self._analysis_job[bad_instruction.name]
                                                              ))
            self.add_message("Can't perform analysis; instructions don't validate.")
            self._stopping = True