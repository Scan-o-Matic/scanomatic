#!/usr/bin/env python
"""
This module is the typical starting-point of the analysis work-flow.
It has command-line behaviour but can also be run as part of another program.
It should be noted that a full run of around 200 images takes more than 2h on
a good computer using 100% of one processor. That is, if run from within
another application, it is probably best to run it as a subprocess.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import sys
import logging
import time
import threading

#
# SCANNOMATIC LIBRARIES
#

import resource_project_log
import resource_analysis_support
import analysis_image
import resource_xml_writer
import resource_path
import resource_app_config
import subprocs.communicator as communicator

#
# GLOBALS
#

#
# FUNCTIONS
#


class Analysis(object):

    def __init__(
            self, log_file_path, outdata_directory, pinning_matrices,
            graph_watch, verbose=False, visual=False,
            #manual_grid=False,
            grid_times=[],
            suppress_analysis=False,
            xml_format={'short': True, 'omit_compartments': [],
                        'omit_measures': []},
            grid_array_settings={'animate': False},
            gridding_settings={'use_otsu': True, 'median_coeff': 0.99,
                               'manual_threshold': 0.05},
            grid_cell_settings={'blob_detect': 'default'},
            grid_correction=None,
            use_local_fixture=True, logger=None, app_config=None,
            comm_id=None):

        """
        analyse_project parses a log-file and runs a full analysis on all
        images in it. It will step backwards in time, starting with the
        last image.

        The function takes the following arguments:

        @log_file_path      The path to the log-file to be processed

        @outdata_files_path  The path to the file were the analysis will
                            be put

        @pinning_matrices   A list/tuple of (row, columns) pinning
                            matrices used for each plate position
                            respectively

        @graph_watch        A coordinate PLATE:COL:ROW for a colony to
                            watch particularly.

        VOID@graph_output   An optional PATH to where to save the graph
                            produced by graph_watch being set.

        @suppress_analysis  Suppresses the main analysis and thus
                            only graph_watch thing is produced.

        @verbose           Will print some basic output of progress.

        @blob_detect        Determines colony detection algorithm
                            (see Grid_Cell_Dissection).

        @grid_otsu          Determines if Otsu-thresholding should be
                            used when looking finding the grid (not
                            default, and should not be used)

        @grid_times         Specifies the time-point indices at which
                            the grids will be saved in the output-dir.

        @xml_format         A dict for what should be in the xml and
                            which tags (long vs short) be used.

        @animate            Boolean def (False) to cause saving animation
                            images.

        @grid_correction    None to use default gridding behaviour, if
                            a tuple of correction shifts for the grid is passed
                            the grid will be moved so many spacings for each
                            dimension.

        The function returns None if nothing was done of if it crashed.
        If it runs through it returns the number of images processed.

        """

        #
        # VARIABLES - SOME ARE HACK
        #

        self._running = None
        self._paused = False
        self._init_time = time.time()
        self._init_fail = False
        self._log_file_path = log_file_path
        self._outdata_directory = outdata_directory
        self._pinning_matrices = pinning_matrices
        self._graph_watch = graph_watch
        self._verbose = verbose
        self._visual = visual
        #self._manual_grid = manual_grid
        self._grid_times = grid_times
        self._suppress_analysis = suppress_analysis
        self._xml_format = xml_format
        self._grid_array_settings = grid_array_settings
        self._gridding_settings = gridding_settings
        self._grid_cell_settings = grid_cell_settings
        self._grid_correction = grid_correction
        self._use_local_fixture = use_local_fixture
        self._app_config = app_config
        self._comm_id = comm_id

        #PATHS AND CONFIG
        self._file_path_base = os.path.abspath(os.path.dirname(log_file_path))
        self._root = os.path.join(self._file_path_base, os.path.pardir)

        self._paths = resource_path.Paths(src_path=__file__)
        if app_config is None:
            app_config = resource_app_config.Config(paths=self._paths)

        self._app_config = app_config

        #
        # VERIFY OUTDATA DIRECTORY
        #

        self._outdata_directory = \
            resource_analysis_support.verify_outdata_directory(outdata_directory)

        #
        # SET UP LOGGER
        #

        hdlr = logging.FileHandler(
            os.path.join(self._outdata_directory, "analysis.run"), mode='w')

        log_formatter = logging.Formatter(
            '\n\n%(asctime)s %(levelname)s:'
            ' %(message)s', datefmt='%Y-%m-%d %H:%M:%S\n')
        hdlr.setFormatter(log_formatter)
        logger.addHandler(hdlr)
        resource_analysis_support.set_logger(logger)

        logger.info('Analysis started at ' + str(self._init_time))
        logger.info('ANALYSIS using file {0}'.format(log_file_path))
        self._logger = logger

        #
        # SET UP EXCEPT HOOK
        #

        sys.excepthook = resource_analysis_support.custom_traceback

        inits = []

        inits.append(self._load_first_pass_file())
        inits.append(self._check_fixture())
        inits.append(self._check_pinning())
        inits.append(self._set_image_dictionary())
        inits.append(self._check_sanity())

        self._init_fail = sum(inits) != len(inits)

        #
        # COMMUNICATIONS TO GUI
        #

        stdin = self._paths.log_analysis_in.format(comm_id)
        stdout = self._paths.log_analysis_out.format(comm_id)
        stderr = self._paths.log_analysis_err.format(comm_id)
        self._comm = communicator.Communicator(
            logger, self,  stdin, stdout, stderr)

        self._comm_thread = threading.Thread(target=self._comm.run)
        self._comm_thread.start()

    def _load_first_pass_file(self):

        logger = self._logger

        #
        # CHECK ANALYSIS-FILE FROM FIRST PASS
        #

        ## META-DATA
        meta_data = resource_project_log.get_meta_data(
            path=self._log_file_path)

        ### METE-DATA BACK COMPATIBILITY
        for key, val in (('Version', 0), ('UUID', None),
                        ('Manual Gridding', None), ('Prefix', ""),
                        ('Project ID', ""), ('Scanner Layout ID', "")):

            if key not in meta_data:
                meta_data[key] = val

        logger.info('ANALYSIS met-data is\n{0}\n'.format(meta_data))

        ### OVERWRITE META-DATA WITH USER INPUT
        if self._pinning_matrices is not None:
            meta_data['Pinning Matrices'] = self._pinning_matrices
            logger.info('ANALYSIS: Pinning matrices use override: {0}'.format(
                self._pinning_matrices))
        if self._use_local_fixture:
            meta_data['Fixture'] = None
            logger.info('ANALYSIS: Local fixture copy to be used')

        self._meta_data = meta_data
        return True

    def _check_fixture(self):

        meta_data = self._meta_data
        logger = self._logger

        #### Test to find Fixture
        if ('Fixture' not in meta_data or
                resource_analysis_support.get_finds_fixture(
                meta_data['Fixture']) is False):

            logger.critical('ANALYSIS: Could not localize fixture settings')
            return False

        return True

    def _check_pinning(self):

        meta_data = self._meta_data
        logger = self._logger

        #### Test if any pinning matrices
        if meta_data['Pinning Matrices'] is None:
            logger.critical(
                "ANALYSIS: need some pinning matrices to analyse anything")
            return False

        return True

    def _set_image_dictionary(self):

        meta_data = self._meta_data
        logger = self._logger

        ## IMAGES
        self._image_dictionaries = resource_project_log.get_image_entries(
            self._log_file_path, logger=logger)

        if len(self._image_dictionaries) == 0:
            logger.critical("ANALYSIS: There are no images to analyse, aborting")

            return False

        logger.info("ANALYSIS: A total of "
                    "{0} images to analyse in project with UUID {1}".format(
                    len(self._image_dictionaries), meta_data['UUID']))

        meta_data['Images'] = len(self._image_dictionaries)
        self._image_pos = meta_data['Images'] - 1

        return True

    def _check_sanity(self):

        #
        # SANITY CHECK
        #

        if resource_analysis_support.get_run_will_do_something(
                self._suppress_analysis,
                self._graph_watch,
                self._meta_data,
                self._image_dictionaries,
                self._logger) is False:

            """
            In principle, if user requests to supress analysis of other
            colonies than the one watched -- then there should be one
            watched and that one needs a pinning matrice.
            """
            return False

        return True

    def run(self):

        self._running = True

        logger = self._logger
        meta_data = self._meta_data
        grid_times = self._grid_times
        image_dictionaries = self._image_dictionaries
        #start_time = time.time()

        #
        # If init produced anything bad, don't run
        #

        if self._init_fail:

            self._comm.set_terminate()
            self._comm_thread.join()
            return False

        #
        # INITIALIZE WATCH GRAPH IF REQUESTED
        #

        if self._graph_watch is not None:

            watch_graph = resource_analysis_support.Watch_Graph(
                self._graph_watch, self._outdata_directory)

        #
        # INITIALIZE XML WRITER
        #

        xml_writer = resource_xml_writer.XML_Writer(
            self._outdata_directory, self._xml_format, logger, self._paths)

        if xml_writer.get_initialized() is False:

            logger.critical('ANALYSIS: XML writer failed to initialize')
            self._comm.set_terminate()
            self._comm_thread.join()
            return False

        #
        # RECORD HOW ANALYSIS WAS STARTED
        #

        logger.info('Analysis was called with the following arguments:\n'
                    'log_file_path\t\t{0}'.format(self._log_file_path) +
                    '\noutdata_file_path\t{0}'.format(
                        self._outdata_directory) +
                    '\nmeta_data\t\t{0}'.format(meta_data) +
                    '\ngraph_watch\t\t{0}'.format(self._graph_watch) +
                    '\nverbose\t\t\t{0}'.format(self._verbose) +
                    '\ngrid_array_settings\t{0}'.format(
                        self._grid_array_settings) +
                    '\ngridding_settings\t{0}'.format(
                        self._gridding_settings) +
                    '\ngrid_cell_settings\t{0}'.format(
                        self._grid_cell_settings) +
                    '\nxml_format\t\t{0}'.format(xml_writer) +
                    '\nmanual_grid\t\t{0}'.format(
                        self._meta_data['Manual Gridding']) +
                    '\ngrid_times\t\t{0}'.format(self._grid_times))

        #
        # GET NUMBER OF PLATES AND THEIR NAMES IN THIS ANALYSIS
        #

        plates, plate_position_keys = resource_analysis_support.get_active_plates(
            meta_data, self._suppress_analysis, self._graph_watch,
            config=self._app_config)

        logger.info('ANALYSIS: These plates ({0}) will be analysed: {1}'.format(
            plates, plate_position_keys))

        if self._suppress_analysis is True:

            meta_data['Pinning Matrices'] = \
                [meta_data['Pinning Matrices'][self._graph_watch[0]]]  # Only keep one

            self._graph_watch[0] = 0  # Since only this one plate is left, it is now 1st

        #
        # INITIALIZING THE IMAGE OBJECT
        #

        project_image = analysis_image.Project_Image(
            meta_data['Pinning Matrices'],
            file_path_base=self._file_path_base,
            fixture_name=meta_data['Fixture'],
            p_uuid=meta_data['UUID'],
            logger=None,  # logger,
            verbose=self._verbose,
            visual=self._visual,
            suppress_analysis=self._suppress_analysis,
            grid_array_settings=self._grid_array_settings,
            gridding_settings=self._gridding_settings,
            grid_cell_settings=self._grid_cell_settings,
            log_version=meta_data['Version'],
            paths=self._paths,
            app_config=self._app_config,
            grid_correction=self._grid_correction
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

        xml_writer.write_header(meta_data, plates)
        xml_writer.write_segment_start_scans()

        #
        # SETTING GRID FROM REASONABLE TIME POINT
        default_gridtime = 217
        if len(grid_times) > 0:
            pos = grid_times[0]
            if pos >= len(image_dictionaries):
                pos = len(image_dictionaries) - 1
        else:
            pos = (len(image_dictionaries) > default_gridtime and default_gridtime
                   or len(image_dictionaries) - 1)

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

        """
        resource_analysis_support.print_progress_bar(size=60)
        """

        while self._image_pos >= 0 and self._running:

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
                    self._outdata_directory,
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
                timestamp=img_dict_pointer['Time'],
                grayscaleTarget=img_dict_pointer['grayscale_indices'],
                image_dict=img_dict_pointer)

            #
            # XML WRITE IT TO FILES
            #

            xml_writer.write_image_features(
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
                "ANALYSIS, Image took {0} seconds".format(
                    (time.time() - scan_start_time)))

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

            #
            # PAUSE IF REQUESTED
            #

            while self._paused and self._running:

                time.sleep(0.5)
        #
        # CLOSING XML TAGS AND FILES
        #

        xml_writer.close()

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
        return True

    def get_current_step(self):

        return self.get_total_iterations() - (self._image_pos + 1)

    def get_total_iterations(self):

        return len(self._image_dictionaries)

    def get_progress(self):

        return float(self.get_current_step()) / self.get_total_iterations()

    def get_paused(self):

        return self._paused

    def set_terminate(self):

        self._running = False
        return True

    def set_pause(self):

        self._paused = True
        return True

    def set_unpause(self):

        self._paused = False
        return True

    def get_running(self):

        return self._running is not False

    def get_info(self):

        return ("__PREFIX__ {0}".format(self._meta_data['Prefix']),
                "__ROOT__ {0}\n".format(self._root),
                "__ANALYSIS DIR__ {0}".format(self._outdata_directory),
                "__1-PASS FILE__ {0}".format(self._log_file_path),
                "__PINNINGS__ {0}".format(self._meta_data['Pinning Matrices']))
