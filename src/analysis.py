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
__version__ = "0.997"
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

#
# GLOBALS
#

#
# FUNCTIONS
#


def analyse_project(log_file_path, outdata_directory, pinning_matrices,
            graph_watch,
            verbose=False, visual=False, manual_grid=False, grid_times=None, 
            suppress_analysis = False,
            xml_format={'short': True, 'omit_compartments': [], 
            'omit_measures': []},
            grid_array_settings = {'animate': False},
            gridding_settings = {'use_otsu': True, 'median_coeff': 0.99,
            'manual_threshold': 0.05},
            grid_cell_settings = {'blob_detect': 'default'},
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

    xml_writer = resource_analysis_support.XML_Writer(outdata_directory,
                    xml_format, logger)

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
                grid_array_settings=grid_array_settings,
                gridding_settings=gridding_settings,
                grid_cell_settings=grid_cell_settings,
                log_version=meta_data['Version']
                )

    # MANUAL GRIDS
    if manual_grid and meta_data['Manual Gridding'] is not None:

        logger.info("ANALYSIS: Will implement manual adjustments of " + \
                    "grid on plates {0}".format(manual_griddings.keys()))
        project_image.set_manual_grids(meta_data['Manual Grid'])

    #
    # WRITING XML HEADERS AND OPENS SCAN TAG
    #

    xml_writer.write_header(meta_data, plates)
    xml_writer.write_segment_start_scans()


    resource_analysis_support.print_progress_bar(size=60)

    #
    # DEBUG BREAK
    #
    xml_writer.close()
    sys.exit()

    while image_pos >= 0:

        scan_start_time = time.time()
        img_dict_pointer = image_dictionaries[image_pos]

        plate_positions = []

        for i in xrange(plates):

            plate_positions.append(
                img_dict_pointer[plate_position_keys[i]])

        logger.info("ANALYSIS, Running analysis on '{}'".format( \
            img_dict_pointer['File']))

        if 'grayscale_indices' in img_dict_pointer.keys():

            gs_indices = img_dict_pointer['grayscale_indices']

        else:

            gs_indices = None

        features = project_image.get_analysis(img_dict_pointer['File'],
            plate_positions, img_dict_pointer['grayscale_values'],
            watch_colony=graph_watch,
            save_grid_name=(image_pos - first_scan_position in grid_times) and
            (outdata_files_path + "time_" + \
            str(image_pos - first_scan_position).zfill(4) + "_plate_") or None,
            identifier_time=image_pos,
            timestamp=img_dict_pointer['Time'],
            grayscale_indices=gs_indices)

        if suppress_analysis != True:

            for f in (fh, fhs):

                f.write(XML_OPEN_W_ONE_PARAM.format(
                            ['scan', 's'][xml_format['short']],
                            ['index', 'i'][xml_format['short']],
                            str(image_pos)))

                f.write(XML_OPEN.format(
                            ['scan-valid', 'ok'][xml_format['short']]))

        if features is None:

            if suppress_analysis != True:

                for f in (fh, fhs):

                    f.write(XML_CONT_CLOSE.format(0,
                            ['scan-valid', 'ok'][xml_format['short']]))

                    f.write(XML_CLOSE.format(
                            ['scan', 's'][xml_format['short']]))

        else:
            if graph_watch != None and  project_image.watch_grid_size != None:

                x_labels.append(image_pos)
                pict_size = project_image.watch_grid_size
                pict_scale = pict_target_width / float(pict_size[1])

                if pict_scale < 1:

                    pict_resize = (int(pict_size[0] * pict_scale),
                                    int(pict_size[1] * pict_scale))

                    plt_watch_1.imshow(Image.fromstring('L',
                            (project_image.watch_scaled.shape[1],
                            project_image.watch_scaled.shape[0]),
                            project_image.watch_scaled.tostring())\
                            .resize(pict_resize, Image.BICUBIC),
                            extent=(image_pos * pict_target_width,
                            (image_pos + 1) * pict_target_width - 1,
                            10, 10 + pict_resize[1]))

                tmp_results = []

                if project_image.watch_results is not None:

                    for cell_item in project_image.watch_results.keys():

                        for measure in project_image.watch_results[\
                                                    cell_item].keys():

                            if type(project_image.watch_results[\
                                            cell_item][measure])\
                                            == np.ndarray or \
                                            project_image.watch_results[\
                                            cell_item][measure] is None:

                                tmp_results.append(np.nan)

                            else:

                                tmp_results.append(
                                        project_image.watch_results[\
                                        cell_item][measure])

                            if len(watch_reading) == 0:

                                plot_labels.append(cell_item + ':' + measure)

                watch_reading.append(tmp_results)

            if suppress_analysis != True:

                for f in (fh, fhs):

                    f.write(XML_CONT_CLOSE.format(1,
                        ['scan-valid>', 'ok'][xml_format['short']]))

                    f.write(XML_OPEN_CONT_CLOSE.format(\
                        ['calibration', 'cal'][xml_format['short']],
                        str(img_dict_pointer['grayscale_values'])))

                    f.write(XML_OPEN_CONT_CLOSE.format(\
                        ['time', 't'][xml_format['short']],
                        str(img_dict_pointer['Time'])))

                    f.write(XML_OPEN.format(
                        ['plates', 'pls'][xml_format['short']]))

                for i in xrange(plates):

                    for f in (fh, fhs):

                        f.write(XML_OPEN_W_ONE_PARAM.format(\
                            ['plate', 'p'][xml_format['short']],
                            ['index', 'i'][xml_format['short']], str(i)))

                        f.write(XML_OPEN_CONT_CLOSE.format(\
                            ['plate-matrix', 'pm'][xml_format['short']],
                            str(pinning_matrices[i])))

                        f.write(XML_OPEN_CONT_CLOSE.format('R',
                            str(project_image.R[i])))

                        f.write(XML_OPEN.format(\
                            ['grid-cells', 'gcs'][xml_format['short']]))

                    for x, rows in enumerate(features[i]):

                        for y, cell in enumerate(rows):

                            for f in (fh, fhs):
                                f.write(XML_OPEN_W_TWO_PARAM.format(\
                                    ['grid-cell', 'gc'][xml_format['short']],
                                    'x', str(x),
                                    'y', str(y)))

                            if cell != None:
                                for item in cell.keys():

                                    i_string = item

                                    if xml_format['short']:

                                        i_string = i_string\
                                                .replace('background', 'bg')\
                                                .replace('blob', 'bl')\
                                                .replace('cell', 'cl')

                                    if item not in \
                                            xml_format['omit_compartments']:

                                        fhs.write(XML_OPEN.format(i_string))

                                    fh.write(XML_OPEN.format(i_string))

                                    for measure in cell[item].keys():

                                        m_string = XML_OPEN_CONT_CLOSE.format(\
                                            str(measure),
                                            str(cell[item][measure]))

                                        if xml_format['short']:

                                            m_string = m_string\
                                                .replace('area', 'a')\
                                                .replace('pixel', 'p')\
                                                .replace('mean', 'm')\
                                                .replace('median', 'md')\
                                                .replace('sum', 's')\
                                                .replace('centroid', 'cent')\
                                                .replace('perimeter', 'per')

                                        if item not in \
                                            xml_format['omit_compartments'] \
                                            and measure not in \
                                            xml_format['omit_measures']:

                                            fhs.write(m_string)

                                        fh.write(m_string)

                                    if item not in \
                                            xml_format['omit_compartments']:

                                        fhs.write(XML_CLOSE.format(i_string))

                                    fh.write(XML_CLOSE.format(i_string))

                            for f in (fh, fhs):

                                f.write(XML_CLOSE.format(\
                                    ['grid-cell', 'gc'][xml_format['short']]))

                    for f in (fh, fhs):

                        f.write(XML_CLOSE.format(\
                            ['grid-cells', 'gcs'][xml_format['short']]))

                        f.write(XML_CLOSE.format(\
                            ['plate', 'p'][xml_format['short']]))

                for f in (fh, fhs):

                    f.write(XML_CLOSE.format(\
                        ['plates', 'pls'][xml_format['short']]))

                    f.write(XML_CLOSE.format(\
                        ['scan', 's'][xml_format['short']]))

        image_pos -= 1

        logger.info("ANALYSIS, Image took %.2f seconds" % \
                         (time.time() - scan_start_time))

        print_progress_bar(
                        fraction = (image_tot - image_pos) / float(image_tot),
                        size=60, start_time=start_time)

    if suppress_analysis != True:

        for f in (fh, fhs):

            f.write(XML_CLOSE.format('scans'))
            f.write(XML_CLOSE.format('project'))

            f.close()

    if  graph_watch != None:

        omits = []
        gws = []

        for i in xrange(len(watch_reading[0])):

            gw_i = [gw[i] for gw in watch_reading]

            try:

                map(lambda v: len(v), gw_i)
                omits.append(i)
                gw_i = None

            except:

                pass

            if gw_i is not None:
                gws.append(gw_i)

        Y = np.asarray(gws, dtype=np.float64)
        X = (np.arange(len(image_dictionaries), 0, -1) + 0.5) *\
                        pict_target_width

        for xlabel_pos in xrange(len(x_labels)):

            if xlabel_pos % 5 > 0:
                x_labels[xlabel_pos] = ""

        cur_plt_graph = ""
        plt_graph_i = 1

        for i in [x for x in range(len(gws)) if x not in omits]:

            ii = i + sum(map(lambda x: x < i, omits))

            Y_good_positions = np.where(np.isnan(Y[i, :]) == False)[0]

            if Y_good_positions.size > 0:

                try:

                    if Y[i, Y_good_positions].max() == \
                                Y[i, Y_good_positions].min():

                        scale_factor = 0

                    else:

                        scale_factor = 100 / \
                                float(Y[i, Y_good_positions].max() - \
                                Y[i, Y_good_positions].min())

                    sub_term = float(Y[i, Y_good_positions].min())

                    if plot_labels[ii] == "cell:area":

                        c_area = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "background:mean":

                        bg_mean = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "cell:pixelsum":

                        c_pixelsum = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "blob:pixelsum":

                        b_pixelsum = Y[i, Y_good_positions]

                    elif plot_labels[ii] == "blob:area":

                        b_area = Y[i, Y_good_positions]

                    logger.debug("WATCH GRAPH:\n%s\n%s\n%s" % \
                                (str(plot_labels[ii]), str(sub_term),
                                str(scale_factor)))

                    logger.debug("WATCH GRAPH, Max %.2f Min %.2f." % \
                            (float(Y[i, Y_good_positions].max()),
                            float(Y[i, Y_good_positions].min())))

                    if cur_plt_graph != plot_labels[ii].split(":")[0]:

                        cur_plt_graph = plot_labels[ii].split(":")[0]

                        if plt_graph_i > 1:

                            plt_watch_curves.legend(loc=1, ncol=5, prop=fontP,
                                    bbox_to_anchor=(1.0, -0.45))

                        plt_graph_i += 1

                        plt_watch_curves = plt_watch_colony.add_subplot(4, 1,
                                    plt_graph_i, title=cur_plt_graph)

                        plt_watch_curves.set_xticks(X)

                        plt_watch_curves.set_xticklabels(x_labels,
                                    fontsize="xx-small", rotation=90)

                    if scale_factor != 0:

                        plt_watch_curves.plot(X[Y_good_positions],
                            (Y[i, Y_good_positions] - sub_term) * scale_factor,
                            label=plot_labels[ii][len(cur_plt_graph) + 1:])

                    else:

                        logger.debug("GRAPH WATCH, Got straight line %s, %s" %\
                                (str(plt_graph_i), str(i)))

                        plt_watch_curves.plot(X[Y_good_positions],
                                np.zeros(X[Y_good_positions].shape) + \
                                10 * (i - (plt_graph_i - 1) * 5),
                                label=plot_labels[ii][len(cur_plt_graph) + 1:])

                except TypeError:

                    logger.warning("GRAPH WATCH, Error processing {0}".format(
                                                            plot_labels[ii]))

            else:

                    logger.warning("GRAPH WATCH, Cann't plot {0}".format(
                            plot_labels[ii]) + "since has no good data.")

        plt_watch_curves.legend(loc=1, ncol=5, prop=fontP,
                            bbox_to_anchor=(1.0, -0.45))

        if graph_output != None:

            try:

                plt_watch_colony.savefig(graph_output, dpi=300)

            except:

                plt_watch_colony.show()

            #DEBUG START:PLOT
            plt_watch_colony = pyplot.figure()
            plt_watch_1 = plt_watch_colony.add_subplot(111)
            plt_watch_1.loglog(b_area, b_pixelsum)
            plt_watch_1.set_xlabel('Blob:Area')
            plt_watch_1.set_ylabel('Blob:PixelSum')
            plt_watch_1.set_title('')
            plt_watch_colony.savefig("debug_corr.png")
            #DEBUG END

            pyplot.close(plt_watch_colony)

        else:

            plt_watch_colony.show()

        logger.info("ANALYSIS, Full analysis took %.2f minutes" %\
            ((time.time() - start_time) / 60.0))

        logger.info('Analysis completed at ' + str(time.time()))

        return False

    logger.info("ANALYSIS, Full analysis took %.2f minutes" %\
        ((time.time() - start_time) / 60.0))

    logger.info('Analysis completed at ' + str(time.time()))

