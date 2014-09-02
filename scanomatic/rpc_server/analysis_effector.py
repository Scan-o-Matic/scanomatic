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

    def __init__(self, identifier, label):

        #sys.excepthook = support.custom_traceback

        super(AnalysisEffector, self).__init__(identifier, label,
                                               loggerName="Analysis Effector")

        self._type = "Analysis"
        self._config = None
        self._startTime = None
        self._metaData = {}
        self._firstImg = None

        self._specificStatuses['progress'] = 'progress'
        self._specificStatuses['total'] = 'totalImages'
        self._specificStatuses['runTime'] = 'runTime'
        self._specificStatuses['currentImage'] = 'curImage'

        self._allowedCalls['setup'] = self.setup

    @property
    def runTime(self):

        if self._startTime is None:
            return 0
        else:
            return time.time() - self._startTime

    @property
    def curImage(self):

        if self._iteratorI is None:
            return -1

        return self._iteratorI

    @property
    def totalImages(self):

        if not self._allowStart or 'Images' not in self._metaData:
            return -1

        return self._metaData['Images']

    @property
    def progress(self):

        total = float(self.totalImages)
        if self._firstImg is None:
            init = 0
        else:
            init = 1

        if total > 0 and self._iteratorI is not None:
            return (init + total - self._iteratorI - 1) / (total + 1)

        return 0

    def next(self):

        if not self._allowStart or not self._running:
            return super(AnalysisEffector, self).next()

        elif self._iteratorI is None:
            self._iteratorI = self.totalImages - 1
            self._startTime = time.time()
            meta_data = self._metaData

            appConfig = app_config.Config()

            #
            # INITIALIZE WATCH GRAPH IF REQUESTED
            #

            if self._watchGraph is not None:

                self._watchGrapher = support.Watch_Graph(
                    self._watchGraph, self._outdataDir)

            #
            # INITIALIZE XML WRITER
            #

            self._xmlWriter = xml_writer.XML_Writer(
                self._outdataDir, self._xmlFormat, paths.Paths())

            if self._xmlWriter.get_initialized() is False:

                self._logger.critical('ANALYSIS: XML writer failed to initialize')
                self._xmlWriter.close()
                self._running = False

                raise StopIteration

            #
            # GET NUMBER OF PLATES AND THEIR NAMES IN THIS ANALYSIS
            #

            self._plates, self._plate_position_keys = support.get_active_plates(
                meta_data, self._suppressAnalysis, self._watchGraph,
                config=appConfig)

            plates = self._plates
            plate_position_keys = self._plate_position_keys

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
                app_config=appConfig,
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
                if pos >= len(self._imageDictionaries):
                    pos = len(self._imageDictionaries) - 1
            else:

                pos = len(self._imageDictionaries) - 1
                """
                pos = (len(self._imageDictionaries) > appConfig.default_gridtime and
                    appConfig.default_gridtime or len(self._imageDictionaries) - 1)
                """

            plate_positions = []

            for i in xrange(plates):

                plate_positions.append(
                    self._imageDictionaries[pos][plate_position_keys[i]])

            self._project_image.set_grid(
                self._imageDictionaries[pos]['File'],
                plate_positions,
                save_name=os.sep.join((
                    self._outdataDir,
                    "grid___origin_plate_")))

            self._firstImg = True
            return True

        elif not self._stopping and self._iteratorI >= 0:

            plates = self._plates
            plate_position_keys = self._plate_position_keys


            #
            # UPDATING LOOP SPECIFIC VARIABLES
            #

            #self._logger.info("__Is__ {0}".format(len(self._imageDictionaries) - self._iteratorI))
            scan_start_time = time.time()
            img_dict_pointer = self._imageDictionaries[self._iteratorI]
            plate_positions = []

            ## PLATE COORDINATES
            for i in xrange(plates):

                plate_positions.append(
                    img_dict_pointer[plate_position_keys[i]])

            ## GRID IMAGE SAVE STRING
            if self._iteratorI in self._gridImageIndices:
                save_grid_name = os.sep.join((
                    self._outdataDir,
                    "grid__time_{0}_plate_".format(str(self._iteratorI).zfill(4))))
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
                img_dict_pointer['File'],
                plate_positions,
                grayscaleSource=img_dict_pointer['grayscale_values'],
                watch_colony=self._watchGraph,
                save_grid_name=save_grid_name,
                identifier_time=self._iteratorI,
                #timestamp=img_dict_pointer['Time'],
                grayscaleTarget=img_dict_pointer['grayscale_indices'],
                image_dict=img_dict_pointer)

            if features is None:
                self._logger.warning("No analysis produced for image")

            #
            # WRITE TO FILES
            #

            image_data.Image_Data.writeTimes(
                self._outdataDir, self._iteratorI, img_dict_pointer,
                overwrite=self._firstImg)
            image_data.Image_Data.writeImage(
                self._outdataDir, self._iteratorI, features, plates)

            self._xmlWriter.write_image_features(
                self._iteratorI, features, img_dict_pointer, plates,
                self._metaData)

            #
            # IF WATCHING A COLONY UPDATE WATCH IMAGE
            #

            if (self._watchGraph is not None and
                    self._project_image.watch_source is not None and
                    self._project_image.watch_blob is not None):

                self._watchGrapher.add_image(
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

            self._iteratorI -= 1
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

        else:

            #
            # CLOSING XML TAGS AND FILES
            #

            self._xmlWriter.close()

            #
            # FINALIZING WATCH GRAPHS
            #

            if  self._watchGraph is not None:

                self._watchGrapher.finalize()

            #
            # OUTPUTS TO USER
            #

            self._logger.info("ANALYSIS, Full analysis took {0} minutes".format(
                ((time.time() - self._startTime) / 60.0)))

            self._logger.info('Analysis completed at ' + str(time.time()))

            self._running = False
            raise StopIteration


    def _safeCfgGet(self, section, item, runKey, defaultValue=None):

        if runKey in self._runInstructions:
            return self._runInstructions[runKey]

        try:

            defaultValue = self._config.get(section, item)

        except:

            pass

        return defaultValue

    def setup(self, *lostArgs, **runInstructions):

        if self._running:
            self.addMessage("Cannot change settings while runnig")

        self._metaData = None

        inits = []
        self._runInstructions = runInstructions
        self._config = ConfigParser(allow_no_value=True)
        if ('configFile' in runInstructions):
            self._config.readfp(open(runInstructions['configFile']))

        self._filePathBase = os.path.dirname(runInstructions['inputFile'])

        inits.append(self._load_first_pass_file(
            runInstructions['inputFile'],
            self._safeCfgGet("Experiment", "pms", "pinningMatrices"),
            self._safeCfgGet("Analysis", "localFixture", "localFixture", 
                                False)))

        self._lastImage = self._safeCfgGet("Analysis", "lastImage", 
                "lastImage", None)

        self._outdataDir = support.verify_outdata_directory(os.path.join(
            self._filePathBase,
            self._safeCfgGet("Analysis", "outputDir",
                "outputDirectory", "analysis")))

        self._watchGraph = self._safeCfgGet("Analysis", "watchPosition",
                "watchPosition")

        self._gridImageIndices = self._safeCfgGet(
                "Analysis", "gridImageIndices", "gridImageIndices", [])

        self._suppressAnalysis = self._safeCfgGet(
            "Analysis", "supressUnwatched", "supressUnwatched", False)

        self._xmlFormat = self._safeCfgGet(
            "Output", "xmlFormat", "xmlFormat", {
                'short': True, 'omit_compartments': [],
                'omit_measures': []})

        self._gridArraySettings = self._safeCfgGet(
            "Analysis", "gridArraySetting", 'gridArraySettings', 
            {'animate': False})

        self._griddingSettings = self._safeCfgGet(
            "Analysis", "griddingSettings", "griddSettings",
            {'use_otsu': True, 'median_coeff': 0.99,
            'manual_threshold': 0.05})

        """
        self._gridTimes = self._safeCfgGet(
            "Analysis", "gridTimes", [])
        """
        
        self._gridCellSettings = self._safeCfgGet(
            "Analysis", "gridCellSettings", "gridCellSettings",
            {'blob_detect': 'default'})

        self._gridCorrection = self._safeCfgGet(
            "Analysis", "gridCorrection", "gridCorrection", None)


        """
        #1st tries to set from explicit path to where project is
        #2nd tries to set from path where first pass file is
        #3rd tries to set from runInstructions path
        self._filePathBase = os.path.abspath(os.path.dirname(
            self._safeCfgGet(
                "Analysis", "basePath", self._safeCfgGet(
                    "First Pass", "path", runInstructions))))
        """

        inits.append(self._check_fixture())
        inits.append(self._check_pinning())
        inits.append(self._set_image_dictionary())
        inits.append(self._check_sanity())

        self._allowStart = sum(inits) == len(inits)
        self._stopping = not self._allowStart

        if not self._allowStart:
            self._logger.error(
                "Can't perform analysis, not all init steps OK: {0}".format(
                    {k: v for k, v in zip(
                        ("First Pass File", "Fixture", "Pinning", "Images",
                         "It all makes sense"), inits)}))

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
        self._imageDictionaries = project_log.get_image_entries(
            self._firstPassFile)
        if self._lastImage is not None:
            self._imageDictionaries[:self._lastImage + 1] #include zero

        if len(self._imageDictionaries) == 0:
            self._logger.critical(
                "ANALYSIS: There are no images to analyse, aborting")

            return False

        self._logger.info(
            "ANALYSIS: A total of " +
            "{0} images to analyse in project with UUID {1}".format(
                len(self._imageDictionaries), meta_data['UUID']))

        meta_data['Images'] = len(self._imageDictionaries)

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
