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
import scanomatic.io.xml.writer as xml_writer
import scanomatic.io.image_data as image_data
import scanomatic.imageAnalysis.support as support
import scanomatic.imageAnalysis.analysis_image as analysis_image
from scanomatic.models.rpc_job_models import JOB_TYPE
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
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

        self._specific_statuses['progress'] = 'progress'
        self._specific_statuses['total'] = 'total'
        self._specific_statuses['current_image_index'] = 'current_image_index'

        self._allowed_calls['setup'] = self.setup

        if job.content_model:
            self._analysis_job = AnalysisModelFactory.create(**job.content_model)
        else:
            self._analysis_job = AnalysisModelFactory.create()
            self._logger.warning("No job instructions")

        self._focus_graph = None
        self._current_image_model = None

    @property
    def current_image_index(self):
        if self._current_image_model:
            return self._current_image_model.index
        return -1

    @property
    def total(self):
        if self._get_is_analysing_images():
            return self._first_pass_results.meta_data.number_of_scans
        return -1

    def _get_is_analysing_images(self):
        return self._allow_start and hasattr(self, "first_pass_results") and self._first_pass_results

    @property
    def progress(self):

        total = float(self.total)
        initiation_weight = 1

        # TODO: Verify this is correct, may underestimate progress
        if total and self._current_image_model:
            return (total - self._current_image_model.index) / float(total + initiation_weight)

        return 0.0

    @property
    def waiting(self):
        return not(self._allow_start and self._running)

    def next(self):
        if self.waiting:
            return super(AnalysisEffector, self).next()
        elif self._current_image_model is None:
            return self._setup_first_iteration()
        elif not self._stopping and self._current_image_model.index > 0:
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
        if not image_model:
            self._stopping = True
            return True

        self._logger.info("ANALYSIS, Running analysis on '{0}'".format(image_model.path))

        features = self._image.get_analysis(image_model)

        if features is None:
            self._logger.warning("No analysis produced for image")

        image_data.Image_Data.writeTimes(self._analysis_job, image_model)
        image_data.Image_Data.writeImage(self._analysis_job, image_model, features)

        self._xmlWriter.write_image_features(image_model, features)

        if self._focus_graph:

            self._focus_graph.add_image(self._image.watch_source, self._image.watch_blob)

        self._logger.info("Image took {0} seconds".format(time.time() - scan_start_time))

        return True

    def _setup_first_iteration(self):

        self._startTime = time.time()

        AnalysisModelFactory.set_absolute_paths(self._analysis_job)

        self._first_pass_results = first_pass_results.FirstPassResults(self._analysis_job.first_pass_file)

        self._remove_files_from_previous_analysis()

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

        self._image = analysis_image.ProjectImage(self._analysis_job, self._first_pass_results.meta_data)

        self._xmlWriter.write_header(self._first_pass_results.meta_data, self._first_pass_results.plates)
        self._xmlWriter.write_segment_start_scans()

        index_for_gridding = self._get_index_for_gridding()

        self._image.set_grid(self._first_pass_results[index_for_gridding])

        return True

    def _remove_files_from_previous_analysis(self):

        for p in image_data.Image_Data.iterImagePaths(self._analysis_job.output_directory):
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

    def setup(self, *args, **kwargs):

        if self._running:
            self.add_message("Cannot change settings while running")
            return

        self._logger.info("Setup got {0} {1}".format(args, kwargs))

        if self._analysis_job.analysis_config_file:
            self._update_job_from_config_file()

        self._allow_start = AnalysisModelFactory.validate(self._analysis_job)

        if not self._allow_start:
            self._logger.error("Can't perform analysis; instructions don't validate.")
            for bad_instruction in AnalysisModelFactory.get_invalid(self._analysis_job):
                self._logger.error("Bad value {0}={1}".format(bad_instruction, self._analysis_job[bad_instruction.name]
                                                              ))
            self.add_message("Can't perform analysis; instructions don't validate.")
            self._stopping = True

    def _update_job_from_config_file(self):

        config = ConfigParser(allow_no_value=True)

        try:
            config.readfp(open(self._analysis_job.analysis_config_file))
        except IOError:
            self._logger.warning("There was no config file at {0}".format(self._analysis_job.analysis_config_file) +
                                 "(if use of analysis config file wasn't intended, please disregard this warning)")
            self._analysis_job.analysis_config_file = None
            return

        AnalysisModelFactory.update(self._analysis_job, **dict(config.items("Analysis")))
        AnalysisModelFactory.update(self._analysis_job, **dict(config.items("Output")))