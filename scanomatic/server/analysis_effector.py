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
import scanomatic.io.project_log as project_log
import scanomatic.io.xml.writer as xml_writer
import scanomatic.io.image_data as image_data
import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.imageAnalysis.support as support
import scanomatic.imageAnalysis.analysis_image as analysis_image
from scanomatic.models.rpc_job_models import JOB_TYPE
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import ImageModel

#
# CLASSES
#


class AnalysisEffector(proc_effector.ProcessEffector):

    TYPE = JOB_TYPE.Analysis

    def __init__(self, job):

        # sys.excepthook = support.custom_traceback

        super(AnalysisEffector, self).__init__(job, logger_name="Analysis Effector")
        self._config = None
        self._metaData = {}
        self._firstImg = None

        self._specific_statuses['progress'] = 'progress'
        self._specific_statuses['total'] = 'total'
        self._specific_statuses['current_image_index'] = 'current_image_index'

        self._allowed_calls['setup'] = self.setup
        self._iteration_index = None
        self._analysis_job = job.content_model

    @property
    def current_image_index(self):

        if self._iteration_index is None:
            return -1

        return self._iteration_index

    @property
    def total(self):

        if not self._allow_start or 'Images' not in self._metaData:
            return -1

        return self._metaData['Images']

    @property
    def progress(self):

        total = float(self.total)
        if self._firstImg is None:
            init = 0
        else:
            init = 1

        if total > 0 and self._iteration_index is not None:
            return (init + total - self._iteration_index - 1) / (total + 1)

        return 0

    def next(self):
        # TODO: Simplify to check if running...
        if not self._allow_start or not self._running:
            return super(AnalysisEffector, self).next()
        elif self._iteration_index is None:
            return self._setup_first_iteration()
        elif not self._stopping and self._iteration_index >= 0:
            return self._analyze_image()
        else:
            return self._finalize_analysis()

    def _finalize_analysis(self):

            self._xmlWriter.close()

            if self._watchGraph is not None:
                self._focus_graph.finalize()


            self._logger.info("ANALYSIS, Full analysis took {0} minutes".format(
                ((time.time() - self._startTime) / 60.0)))

            self._logger.info('Analysis completed at ' + str(time.time()))

            self._running = False
            raise StopIteration

    def _analyze_image(self):

        plates = self._plates
        plate_position_keys = self._plate_position_keys

        #
        # UPDATING LOOP SPECIFIC VARIABLES
        #

        scan_start_time = time.time()
        img_dict_pointer = self._image_models[self._iteration_index]
        image_model = ImageModel()
        plate_positions = []

        # PLATE COORDINATES
        for i in xrange(plates):

            plate_positions.append(
                img_dict_pointer[plate_position_keys[i]])

        # GRID IMAGE SAVE STRING
        if self._iteration_index in self._gridImageIndices:
            save_grid_name = os.sep.join((
                self._outdataDir,
                "grid__time_{0}_plate_".format(str(self._iteration_index).zfill(4))))
        else:
            save_grid_name = None

        #
        # INFO TO USER
        #

        self._logger.info("ANALYSIS, Running analysis on '{}'".format(
            img_dict_pointer['File']))

        #
        # GET ANALYSIS
        #

        features = self._project_image.get_analysis(
            image_model.path,
            plate_positions,
            grayscaleSource=image_model.grayscale_values,
            save_grid_name=save_grid_name,
            identifier_time=self._iteration_index,
            grayscaleTarget=image_model.grayscale_targets,
            image_dict=img_dict_pointer)

        if features is None:
            self._logger.warning("No analysis produced for image")

        #
        # WRITE TO FILES
        #

        image_data.Image_Data.writeTimes(
            self._outdataDir, self._iteration_index, img_dict_pointer,
            overwrite=self._firstImg)
        image_data.Image_Data.writeImage(
            self._outdataDir, self._iteration_index, features, plates)

        self._xmlWriter.write_image_features(
            self._iteration_index, features, img_dict_pointer, plates,
            self._metaData)

        #
        # IF WATCHING A COLONY UPDATE WATCH IMAGE
        #

        if (self._watchGraph is not None and
                self._project_image.watch_source is not None and
                self._project_image.watch_blob is not None):

            self._focus_graph.add_image(
                self._project_image.watch_source,
                self._project_image.watch_blob)

        #
        # USER INFO
        #

        self._logger.info(
            "Image took {0} seconds".format(
                (time.time() - scan_start_time)))

        """
        for handler in self._logger.handlers:
            handler.flush()
        """

        """
        resource_analysis_support.print_progress_bar(
            fraction=(self._metaData['Images'] - self._iteratorI) /
            float(self._metaData['Images']),
            size=60, start_time=start_time)
        """
        #
        # UPDATE IMAGE_POS
        #

        self._iteration_index -= 1
        self._firstImg = False

        """
        #DEBUGGING memory
        print "--"
        objgraph.show_growth(limit=20)

        for dbgI in range(40):
            objgraph.show_chain(
                objgraph.find_backref_chain(
                    random.choice(objgraph.by_type(
                        'list')), inspect.ismodule),
                filename='memDebug{0}.{1}.png'.format(
                    str(self._iteratorI).zfill(4), dbgI))

                #self._project_image[0][(0, 0)].blob.filter_array, inspect.ismodule),

            print ">im", dbgI
        random.choice(objgraph.by_type(
            'instance')),
        """

        return True

    def _setup_first_iteration(self):

        # TODO: Make sure iteration is done on time and not index in 1st pass file
        self._iteration_index = self.total - 1
        self._startTime = time.time()
        meta_data = self._metaData

        application_config = app_config.Config()

        #
        # CLEANING UP PREVIOUS FILES
        #

        # TODO: Need to convert to absolute
        for p in image_data.Image_Data.iterImagePaths(self._analysis_job.output_directory):
            os.remove(p)

        self._loger.info("Removed pre-existing file '{0}'".format(p))
        #
        # INITIALIZE WATCH GRAPH IF REQUESTED
        #

        if self._analysis_job.focus_position is not None:

            self._focus_graph = support.Watch_Graph(
                self._analysis_job.focus_position, self._analysis_job.output_directory)

        #
        # INITIALIZE XML WRITER
        #

        self._xmlWriter = xml_writer.XML_Writer(
            self._analysis_job.output_directory, self._analysis_job.xml_model)

        if self._xmlWriter.get_initialized() is False:

            self._logger.critical('ANALYSIS: XML writer failed to initialize')
            self._xmlWriter.close()
            self._running = False

            raise StopIteration

        #
        # GET NUMBER OF PLATES AND THEIR NAMES IN THIS ANALYSIS
        #

        self._plates, self._plate_position_keys = support.get_active_plates(
            meta_data, self._analysis_job.suppress_non_focal, self._analysis_job.focus_position,
            config=application_config)

        plates = self._plates
        plate_position_keys = self._plate_position_keys

        self._logger.info(
            'ANALYSIS: These plates ({0}) will be analysed: {1}'.format(
                plates, plate_position_keys))

        if self._analysis_job.suppress_non_focal is True:

            meta_data['Pinning Matrices'] = [meta_data['Pinning Matrices'][self._watchGraph[0]]]
            self._watchGraph[0] = 0  # Since only this one plate is left, it is now 1st

        #
        # INITIALIZING THE IMAGE OBJECT
        #

        self._project_image = analysis_image.Project_Image(
            meta_data['Pinning Matrices'],
            file_path_base=self._filePathBase,
            fixture_name=meta_data['Fixture'],
            p_uuid=meta_data['UUID'],
            suppress_analysis=self._suppressAnalysis,
            grid_array_settings=self._gridArraySettings,
            gridding_settings=self._griddingSettings,
            grid_cell_settings=self._gridCellSettings,
            log_version=meta_data['Version'],
            app_config=application_config,
            grid_correction=self._gridCorrection
        )

        '''
        # MANUAL GRIDS
        if self._manual_grid and meta_data['Manual Gridding'] is not None:

            self._logger.info("ANALYSIS: Will implement manual adjustments of "
                        "grid on plates {0}".format(meta_data['Maunual Gridding'].keys()))

            self._project_image.set_manual_ideal_grids(meta_data['Manual Grid'])
        '''

        #
        # WRITING XML HEADERS AND OPENS SCAN TAG
        #

        self._xmlWriter.write_header(meta_data, plates)
        self._xmlWriter.write_segment_start_scans()

        #
        # SETTING GRID FROM REASONABLE TIME POINT
        if len(self._gridImageIndices) > 0:
            pos = max(self._gridImageIndices)
            if pos >= len(self._image_models):
                pos = len(self._image_models) - 1
        else:

            pos = len(self._image_models) - 1
            """
            pos = (len(self._imageDictionaries) > application_config.default_gridtime and
                application_config.default_gridtime or len(self._imageDictionaries) - 1)
            """

        plate_positions = []

        for i in xrange(plates):

            plate_positions.append(
                self._image_models[pos][plate_position_keys[i]])

        self._project_image.set_grid(
            self._image_models[pos]['File'],
            plate_positions,
            save_name=os.sep.join((
                self._outdataDir,
                "grid___origin_plate_")))

        self._firstImg = True
        return True

    def setup(self, *args, **kwargs):

        if self._running:
            self.add_message("Cannot change settings while running")

        self._metaData = None

        if self._job.analysis_config_file:
            self._update_job_from_config_file()

        self._allow_start = AnalysisModelFactory.validate(self._analysis_job)

        if not self._allow_start:
            self._logger.error("Can't perform analysis; instructions don't validate.")
            self.add_message("Can't perform analysis; instructions don't validate.")

    def _update_job_from_config_file(self):

        config = ConfigParser(allow_no_value=True)
        config.readfp(open(self.))
        AnalysisModelFactory.update(self._analysis_job, **dict(config.items("Analysis")))
        AnalysisModelFactory.update(self._analysis_job, **dict(config.items("Output")))

    def _load_first_pass_file(self, path, pms, localFixture):

        #
        # CHECK ANALYSIS-FILE FROM FIRST PASS
        #

        if not os.path.isfile(path):
            return False

        self._firstPassFile = path

        ## META-DATA
        meta_data = project_log.get_meta_data(
            path=path)

        ### METE-DATA BACK COMPATIBILITY
        for key, val in (('Version', 0), ('UUID', None),
                        ('Manual Gridding', None), ('Prefix', ""),
                        ('Project ID', ""), ('Scanner Layout ID', "")):

            if key not in meta_data:
                meta_data[key] = val

        ### OVERWRITE META-DATA WITH USER INPUT
        if pms is not None:
            meta_data['Pinning Matrices'] = pms
            self._logger.info(
                'Pinning matrices overridden with: {0}'.format(
                    pms))

        if localFixture:
            meta_data['Fixture'] = None
            self._logger.info('Local fixture copy to be used')

        self._metaData = meta_data

        return True

    def _check_fixture(self):

        meta_data = self._metaData

        #### Test to find Fixture
        if ('Fixture' not in meta_data or
                support.get_finds_fixture(meta_data['Fixture']) is False):

            self._logger.critical(
                'ANALYSIS: Could not localize fixture settings')
            return False

        return True

    def _check_pinning(self):

        meta_data = self._metaData

        #### Test if any pinning matrices
        if meta_data['Pinning Matrices'] is None:
            self._logger.critical(
                "ANALYSIS: need some pinning matrices to analyse anything")
            return False

        return True

    def _set_image_dictionary(self):

        meta_data = self._metaData

        ## IMAGES
        self._image_models = project_log.get_image_entries(self._firstPassFile)

        if self._lastImage is not None:
            self._image_models[:self._lastImage + 1] #include zero

        if len(self._image_models) == 0:
            self._logger.critical(
                "ANALYSIS: There are no images to analyse, aborting")

            return False

        self._logger.info(
            "ANALYSIS: A total of " +
            "{0} images to analyse in project with UUID {1}".format(
                len(self._image_models), meta_data['UUID']))

        meta_data['Images'] = len(self._image_models)

        return True