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
import scanomatic.io.logger as logger
import scanomatic.io.project_log as project_log
import scanomatic.io.xml.writer as xml_writer
import scanomatic.io.image_data as image_data
import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.imageAnalysis.support as support
import scanomatic.imageAnalysis.analysis_image as analysis_image

#
# CLASSES
#


class AnalysisEffector(proc_effector.ProcEffector):

    def __init__(self):

        #sys.excepthook = support.custom_traceback

        super(AnalysisEffector, self).__init__(loggerName="Analysis Effector")

        self._curImageId = -1

        self._specificStatuses['progress'] = 'progress'
        self._specificStatuses['total'] = 'totalImages'
        self._specificStatuses['startTime'] = '_startTime'

        self._allowedCalls['setup'] = self.setup
        self._config = None

        self._start_time = None

    @property
    def totalImages(self):

        if not self._allowStart:
            return -1

    @property
    def progress(self):

        total = self.totalImages
        if total > 0 and self._curImageId > 0:
            return total - self._curImageId - 1

    def next(self):

        super(AnalysisEffector, self).next()

        self._start_time = time.time()

        meta_data = self._metaData
        appConfig = app_config.App_Config()
        image_dictionaries = self._imageDictionaries
        grid_times = self._gridTimes

        #
        # INITIALIZE WATCH GRAPH IF REQUESTED
        #

        if self._watchGraph is not None:

            watch_graph = support.Watch_Graph(
                self._watchGraph, self._outdataDir)

        #
        # INITIALIZE XML WRITER
        #

        xmlWriter = xml_writer.XML_Writer(
            self._outdataDir, self._xmlFormat, paths.Paths())

        if xmlWriter.get_initialized() is False:

            self._logger.critical('ANALYSIS: XML writer failed to initialize')
            self._running = False

        #
        # GET NUMBER OF PLATES AND THEIR NAMES IN THIS ANALYSIS
        #

        plates, plate_position_keys = support.get_active_plates(
            meta_data, self._suppressAnalysis, self._watchGraph,
            config=appConfig)

        self._logger.info(
            'ANALYSIS: These plates ({0}) will be analysed: {1}'.format(
                plates, plate_position_keys))

        if self._suppressAnalysis is True:

            meta_data['Pinning Matrices'] = \
                [meta_data['Pinning Matrices'][self._watchGraph[0]]]  # Only keep one

            self._watchGraph[0] = 0  # Since only this one plate is left, it is now 1st

        #
        # INITIALIZING THE IMAGE OBJECT
        #

        project_image = analysis_image.Project_Image(
            meta_data['Pinning Matrices'],
            file_path_base=self._filePathBase,
            fixture_name=meta_data['Fixture'],
            p_uuid=meta_data['UUID'],
            suppress_analysis=self._suppressAnalysis,
            grid_array_settings=self._gridArraySettings,
            gridding_settings=self._griddingSettings,
            grid_cell_settings=self._gridCellSettings,
            log_version=meta_data['Version'],
            app_config=appConfig,
            grid_correction=self._gridCorrection
        )

        '''
        # MANUAL GRIDS
        if self._manual_grid and meta_data['Manual Gridding'] is not None:

            logger.info("ANALYSIS: Will implement manual adjustments of "
                        "grid on plates {0}".format(meta_data['Maunual Gridding'].keys()))

            project_image.set_manual_ideal_grids(meta_data['Manual Grid'])
        '''

        #
        # WRITING XML HEADERS AND OPENS SCAN TAG
        #

        xmlWriter.write_header(meta_data, plates)
        xmlWriter.write_segment_start_scans()

        #
        # SETTING GRID FROM REASONABLE TIME POINT
        if len(grid_times) > 0:
            pos = grid_times[0]
            if pos >= len(image_dictionaries):
                pos = len(image_dictionaries) - 1
        else:

            pos = (len(image_dictionaries) > appConfig.default_gridtime and
                   appConfig.default_gridtime or len(image_dictionaries) - 1)

        plate_positions = []

        for i in xrange(plates):

            plate_positions.append(
                image_dictionaries[pos][plate_position_keys[i]])

        project_image.set_grid(
            image_dictionaries[pos]['File'],
            plate_positions,
            save_name=os.sep.join((
                self._outdata_directory,
                "grid___origin_plate_")))

        firstImg = True

        while self._running:

            #
            # UPDATING LOOP SPECIFIC VARIABLES
            #

            #logger.info("__Is__ {0}".format(len(image_dictionaries) - self._image_pos))
            scan_start_time = time.time()
            img_dict_pointer = image_dictionaries[self._image_pos]
            plate_positions = []

            ## PLATE COORDINATES
            for i in xrange(plates):

                plate_positions.append(
                    img_dict_pointer[plate_position_keys[i]])

            ## GRID IMAGE SAVE STRING
            if self._image_pos in grid_times:
                save_grid_name = os.sep.join((
                    self._outdataDir,
                    "grid__time_{0}_plate_".format(str(self._image_pos).zfill(4))))
            else:
                save_grid_name = None

            #
            # INFO TO USER
            #

            logger.info("ANALYSIS, Running analysis on '{}'".format(
                img_dict_pointer['File']))

            #
            # GET ANALYSIS
            #

            features = project_image.get_analysis(
                img_dict_pointer['File'],
                plate_positions,
                grayscaleSource=img_dict_pointer['grayscale_values'],
                watch_colony=self._graph_watch,
                save_grid_name=save_grid_name,
                identifier_time=self._image_pos,
                #timestamp=img_dict_pointer['Time'],
                grayscaleTarget=img_dict_pointer['grayscale_indices'],
                image_dict=img_dict_pointer)

            if features is None:
                logger.warning("No analysis produced for image")

            #
            # WRITE TO FILES
            #

            image_data.Image_Data.writeTimes(
                self._outdataDir, self._image_pos, img_dict_pointer,
                overwrite=firstImg)
            image_data.Image_Data.writeImage(
                self._outdataDir, self._image_pos, features, plates)

            xmlWriter.write_image_features(
                self._image_pos, features, img_dict_pointer, plates, meta_data)

            #
            # IF WATCHING A COLONY UPDATE WATCH IMAGE
            #

            if (self._graph_watch is not None and
                    project_image.watch_source is not None and
                    project_image.watch_blob is not None):

                watch_graph.add_image(
                    project_image.watch_source,
                    project_image.watch_blob)

            #
            # USER INFO
            #

            logger.info(
                "Image took {0} seconds".format(
                    (time.time() - scan_start_time)))

            """
            for handler in logger.handlers:
                handler.flush()
            """

            """
            resource_analysis_support.print_progress_bar(
                fraction=(meta_data['Images'] - self._image_pos) /
                float(meta_data['Images']),
                size=60, start_time=start_time)
            """
            #
            # UPDATE IMAGE_POS
            #

            self._image_pos -= 1
            firstImg = False

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
                        str(self._image_pos).zfill(4), dbgI))

                    #project_image[0][(0, 0)].blob.filter_array, inspect.ismodule),

                print ">im", dbgI
            random.choice(objgraph.by_type(
                'instance')),
            """

            yield

            #
            # PAUSE IF REQUESTED
            #

            while self._paused and self._running:

                time.sleep(0.5)
                yield

        #
        # CLOSING XML TAGS AND FILES
        #

        xmlWriter.close()

        #
        # FINALIZING WATCH GRAPHS
        #

        if  self._graph_watch is not None:

            watch_graph.finalize()

        #
        # OUTPUTS TO USER
        #

        logger.info("ANALYSIS, Full analysis took {0} minutes".format(
            ((time.time() - self._init_time) / 60.0)))

        logger.info('Analysis completed at ' + str(time.time()))

        self._running = False
        self._comm.set_terminate()
        self._comm_thread.join()

        logger.info("Exiting")

    def _safeCfgGet(self, section, item, defaultValue=None):

        try:

            defaultValue = self._config.get(section, item)

        except:

            pass

        return defaultValue

    def setup(self, runInstructions=None):

        if self._running:
            self.addMessage("Cannot change settings while runnig")

        self._metaData = None

        if runInstructions is not None and os.path.isfile(runInstructions):

            inits = []

            self._config = ConfigParser(allow_no_values=True)
            self._config.readfp(open(runInstructions))

            inits.append(self._load_first_pass_file(
                self._safeCfgGet("First Pass", "path"),
                self._safeCfgGet("Experiment", "pms"),
                self._safeCfgGet("Analysis", "localFixture", False)))

            self._curImageId = self._safeCfgGet("Analysis", "startIndex", 0)

            self._outdataDir = support.verify_outdata_directory(
                self._safeCfgGet("Analysis", "outputDir", "analysis"))

            self._watchGraph = self._safeCfgGet("Analysis", "watchPosition")

            self._gridImageIndices = self._safeCfgGet("Analysis",
                                                      "gridImageIndices", [])

            self._suppressAnalysis = self._safeCfgGet("Analysis",
                                                      "suppress", False)

            self._xmlFormat = self._safeCfgGet(
                "Output", "xmlFormat", {
                    'short': True, 'omit_compartments': [],
                    'omit_measures': []})

            self._gridArraySettings = self._safeCfgGet(
                "Analysis", "gridArraySetting", {'animate': False})

            self._griddingSettings = self._safeCfgGet(
                "Analysis", "griddingSettings",
                {'use_otsu': True, 'median_coeff': 0.99,
                'manual_threshold': 0.05})

            self._gridTimes = self._safeCfgGet(
                "Analysis", "gridTimes", [])

            self._gridCellSettings = self._safeCfgGet(
                "Analysis", "gridCellSettings",
                {'blob_detect': 'default'})

            self._gridCorrection = self._safeCfgGet(
                "Analysis", "gridCorrection", None)

            #1st tries to set from explicit path to where project is
            #2nd tries to set from path where first pass file is
            #3rd tries to set from runInstructions path
            self._filePathBase = os.path.abspath(os.path.dirname(
                self._safeCfgGet(
                    "Analysis", "basePath", self._safeCfgGet(
                        "First Pass", "path", runInstructions))))

            inits.append(self._check_fixture())
            inits.append(self._check_pinning())
            inits.append(self._set_image_dictionary())
            inits.append(self._check_sanity())

            self._allowStart = sum(inits) == len(inits)

    def _load_first_pass_file(self, path, pms, localFixture):

        #
        # CHECK ANALYSIS-FILE FROM FIRST PASS
        #

        if not os.path.isfile(path):
            return False

        self._firstPassFile = path

        ## META-DATA
        meta_data = project_log.get_metaData(
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
        self._imageDictionaries = project_log.get_image_entries(
            self._firstPassFile)

        if len(self._image_dictionaries) == 0:
            self._logger.critical(
                "ANALYSIS: There are no images to analyse, aborting")

            return False

        self._logger.info(
            "ANALYSIS: A total of " +
            "{0} images to analyse in project with UUID {1}".format(
                len(self._imageDictionaries), meta_data['UUID']))

        meta_data['Images'] = len(self._imageDictionaries)
        self._curImageId = meta_data['Images'] - 1

        return True

    def _check_sanity(self):

        #
        # SANITY CHECK
        #

        if support.get_run_will_do_something(
                self._suppressAnalysis,
                self._watchGraph,
                self._metaData,
                self._imageDictionaries) is False:

            """
            In principle, if user requests to supress analysis of other
            colonies than the one watched -- then there should be one
            watched and that one needs a pinning matrice.
            """
            return False

        return True
