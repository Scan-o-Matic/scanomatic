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
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
import logging
import numpy as np
import time

#
# SCANNOMATIC LIBRARIES
#

import resource_project_log
import resource_analysis_support
import analysis_image
import resource_xml_writer
import resource_path

#
# GLOBALS
#

#
# FUNCTIONS
#


def analyse_project(log_file_path, outdata_directory, pinning_matrices,
            graph_watch,
            verbose=False, visual=False, manual_grid=False, grid_times=[], 
            suppress_analysis = False,
            xml_format={'short': True, 'omit_compartments': [], 
            'omit_measures': []},
            grid_array_settings = {'animate': False},
            gridding_settings = {'use_otsu': True, 'median_coeff': 0.99,
            'manual_threshold': 0.05},
            grid_cell_settings = {'blob_detect': 'default'},
            use_local_fixture=True,
            logger=None):
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

        The function returns None if nothing was done of if it crashed.
        If it runs through it returns the number of images processed.

    """

    #
    # VARIABLES - SOME ARE HACK
    #

    start_time = time.time()
    graph_output = None
    file_path_base = os.sep.join(log_file_path.split(os.sep)[:-1])
    #grid_adjustments = None

    paths = resource_path.Paths(src_path=__file__)

    #
    # VERIFY OUTDATA DIRECTORY
    #

    outdata_directory = \
        resource_analysis_support.verify_outdata_directory(outdata_directory)

    #
    # SET UP LOGGER
    #

    hdlr = logging.FileHandler(
        os.sep.join((outdata_directory, "analysis.run")), mode='w')
    log_formatter = logging.Formatter('\n\n%(asctime)s %(levelname)s:' + \
                    ' %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S\n')
    hdlr.setFormatter(log_formatter)
    logger.addHandler(hdlr)
    resource_analysis_support.set_logger(logger)

    logger.info('Analysis started at ' + str(start_time))

    #
    # SET UP EXCEPT HOOK
    #

    sys.excepthook = resource_analysis_support.custom_traceback

    #
    # CHECK ANALYSIS-FILE FROM FIRST PASS
    #

    ## META-DATA
    meta_data = resource_project_log.get_meta_data(path=log_file_path)
    
    ### METE-DATA BACK COMPATIBILITY
    if 'Version' not in meta_data:
        meta_data['Version'] = 0
    if 'UUID' not in meta_data:
        meta_data['UUID'] = None
    if 'Manual Gridding' not in meta_data:
        meta_data['Manual Gridding'] = None

    ### OVERWRITE META-DATA WITH USER INPUT
    if pinning_matrices is not None:
        meta_data['Pinning Matrices'] = pinning_matrices
        logger.info('ANALYSIS: Pinning matrices use override: {0}'.format(
            pinning_matrices))
    if use_local_fixture:
        meta_data['Fixture'] = None
        logger.info('ANALYSIS: Local fixture copy to be used')

    ### VERIFYING VITAL ASPECTS

    #### Test to find Fixture
    if 'Fixture' not in meta_data or \
        resource_analysis_support.get_finds_fixture(
        meta_data['Fixture']) == False:

        logger.critical('ANALYSIS: Could not localize fixture settings')
        return False

    #### Test if any pinning matrices
    if meta_data['Pinning Matrices'] is None:
        logger.critical(
            "ANALYSIS: need some pinning matrices to analyse anything")
        return False

    ## IMAGES
    image_dictionaries = resource_project_log.get_image_entries(log_file_path)

    if len(image_dictionaries) == 0:
        logger.critical("ANALYSIS: There are no images to analyse, aborting")

        return False

    logger.info("ANALYSIS: A total of " + 
                    "{0} images to analyse in project with UUID {1}".format(
                    len(image_dictionaries),
                    meta_data['UUID']))

    meta_data['Images'] = len(image_dictionaries)
    image_pos = meta_data['Images'] - 1

    #
    # SANITY CHECK
    #

    if resource_analysis_support.get_run_will_do_something(
            suppress_analysis, graph_watch, meta_data, logger) == False:

        """
        In principle, if user requests to supress analysis of other
        colonies than the one watched -- then there should be one
        watched and that one needs a pinning matrice.
        """
        return False

    #
    # INITIALIZE WATCH GRAPH IF REQUESTED
    #

    if graph_watch != None:

        watch_graph = resource_analysis_support.Watch_Graph(graph_watch,
                                outdata_directory)

    #
    # INITIALIZE XML WRITER
    #

    xml_writer = resource_xml_writer.XML_Writer(outdata_directory,
                    xml_format, logger, paths)

    if xml_writer.get_initialized() == False:
        logger.critical('ANALYSIS: XML writer failed to initialize')
        return False

    #
    # RECORD HOW ANALYSIS WAS STARTED
    #

    logger.info('Analysis was called with the following arguments:\n' +
        'log_file_path\t\t{0}'.format(log_file_path) + 
        '\noutdata_file_path\t{0}'.format(outdata_directory) + 
        '\nmeta_data\t\t{0}'.format(meta_data) +
        '\ngraph_watch\t\t{0}'.format(graph_watch) +
        '\nverbose\t\t\t{0}'.format(verbose) + 
        '\ngrid_array_settings\t{0}'.format(grid_array_settings) + 
        '\ngridding_settings\t{0}'.format(gridding_settings) + 
        '\ngrid_cell_settings\t{0}'.format(grid_cell_settings) +
        '\nxml_format\t\t{0}'.format(xml_writer) +
        '\nmanual_grid\t\t{0}'.format(meta_data['Manual Gridding']) +
        '\ngrid_times\t\t{0}'.format(grid_times))

    #
    # GET NUMBER OF PLATES AND THEIR NAMES IN THIS ANALYSIS
    #

    plates, plate_position_keys = resource_analysis_support.get_active_plates(
        meta_data, suppress_analysis, graph_watch)

    logger.info('ANALYSIS: These plates ({0}) will be analysed: {1}'.format(
        plates, plate_position_keys))

    if suppress_analysis == True:

        meta_data['Pinning Matrices'] = \
            [meta_data['Pinning Matrices'][graph_watch[0]]] # Only keep one

        graph_watch[0] = 0  # Since only this one plate is left, it is now 1st

    #
    # INITIALIZING THE IMAGE OBJECT
    #

    project_image = analysis_image.Project_Image(
                meta_data['Pinning Matrices'],
                file_path_base=file_path_base,
                fixture_name=meta_data['Fixture'],
                p_uuid=meta_data['UUID'],
                logger=None,
                verbose=verbose,
                visual=visual,
                suppress_analysis=suppress_analysis,
                grid_array_settings=grid_array_settings,
                gridding_settings=gridding_settings,
                grid_cell_settings=grid_cell_settings,
                log_version=meta_data['Version']
                )

    # MANUAL GRIDS
    if manual_grid and meta_data['Manual Gridding'] is not None:

        logger.info("ANALYSIS: Will implement manual adjustments of " + \
                    "grid on plates {0}".format(manual_griddings.keys()))
        project_image.set_manual_ideal_grids(meta_data['Manual Grid'])

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

    project_image.set_grid(image_dictionaries[pos]['File'],
        plate_positions,
        save_name = os.sep.join((
        outdata_directory,
        "grid___origin_plate_")))
 

    resource_analysis_support.print_progress_bar(size=60)

    while image_pos >= 0:

        #
        # UPDATING LOOP SPECIFIC VARIABLES
        #

        logger.info("__Is__ {0}".format(len(image_dictionaries) - image_pos))
        scan_start_time = time.time()
        img_dict_pointer = image_dictionaries[image_pos]
        plate_positions = []

        ## PLATE COORDINATES
        for i in xrange(plates):

            plate_positions.append(
                img_dict_pointer[plate_position_keys[i]])

        ## GRID IMAGE SAVE STRING
        if image_pos in grid_times:
            save_grid_name = os.sep.join((
                outdata_directory,
                "grid__time_{0}_plate_".format(str(image_pos).zfill(4))))
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

        features = project_image.get_analysis(img_dict_pointer['File'],
            plate_positions, img_dict_pointer['grayscale_values'],
            watch_colony=graph_watch,
            save_grid_name=save_grid_name,
            identifier_time=image_pos,
            timestamp=img_dict_pointer['Time'],
            grayscale_indices=img_dict_pointer['grayscale_indices'],
            image_dict=img_dict_pointer)

        #
        # XML WRITE IT TO FILES
        #

        xml_writer.write_image_features(image_pos, features, img_dict_pointer,
            plates, meta_data)

        #
        # IF WATCHING A COLONY UPDATE WATCH IMAGE
        #

        if graph_watch is not None:

            watch_graph.add_image()

        #
        # USER INFO
        #

        logger.info("ANALYSIS, Image took %.2f seconds" % \
                         (time.time() - scan_start_time))

        resource_analysis_support.print_progress_bar(
                        fraction = (meta_data['Images'] - image_pos) / \
                        float(meta_data['Images']),
                        size=60, start_time=start_time)

        #
        # UPDATE IMAGE_POS
        #

        image_pos -= 1


    #
    # CLOSING XML TAGS AND FILES
    #

    xml_writer.close()

    #
    # FINALIZING WATCH GRAPHS
    #

    if  graph_watch is not None:

        watch_graph.finalize()

    #
    # OUTPUTS TO USER
    #

    logger.info("ANALYSIS, Full analysis took %.2f minutes" %\
        ((time.time() - start_time) / 60.0))

    logger.info('Analysis completed at ' + str(time.time()))

