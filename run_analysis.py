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

import sys
import os
import logging
from argparse import ArgumentParser

#
# INTERNAL DEPENDENCIES
#

import src.analysis as analysis
import src.resource_analysis_support as resource_analysis_support

#
# RUN BEHAVIOUR
#

if __name__ == "__main__":

    print "ANYALYSIS WAS CALLED WITH:"
    print " ".join(sys.argv)
    print
    print 

    parser = ArgumentParser(description='The analysis script runs through " +\
                "a log-file (which is created when a project is run). It " + \
                "creates a XML-file that holds the result of the analysis')

    parser.add_argument("-i", "--input-file", type=str, dest="inputfile",
        help="Log-file to be parsed", metavar="PATH")

    parser.add_argument("-o", "--ouput-path", type=str, dest="outputpath",
        help="Path to directory where all data is written (Default is a " +\
        "subdirectory 'analysis' under where the input file is)",
        metavar="PATH")

    parser.add_argument("-m", "--matrices", dest="matrices", default=None,
        help="The pinning matrices for each plate position in the order " + \
        "set by the fixture config", metavar="(X,Y):(X,Y)...(X,Y)")

    parser.add_argument("-w", "--watch-position", dest="graph_watch",
        help="The position of a colony to track.", metavar="PLATE:X:Y",
        type=str)

    parser.add_argument("-t", "--watch-time", dest="grid_times",
        help="If specified, the gridplacements at the specified timepoints" + \
        " will be saved in the set output-directory, comma-separeted indices.",
        metavar="0,1,100", default="0", type=str)

    parser.add_argument("-g", "--manual-grid", dest="manual_grid",
        help="Boolean used to invoke manually set gridding, default is false",
        default=False, type=bool)

    parser.add_argument('-a', '--animate', dest="animate", default=False,
        type=bool, help="If True, it will produce stop motion images of the" +\
        "watched colony ready for animation")

    parser.add_argument("--grid-otsu", dest="g_otsu", default=False, type=bool,
        help="Invokes the usage of utso segmentation for detecting the grid")

    parser.add_argument("--blob-detection", dest="b_detect", default="default",
        type=str,
        help="Determines which algorithm will be used to detect blobs." +
        "Currently, only 'default'")

    parser.add_argument("-s", "--suppress-analysis", dest="suppress",
        default=False, type=bool,
        help="If submitted, main analysis will be by-passed and only the" + \
        " plate and position that was specified by the -w flag will be " + \
        "analysed and reported.")

    parser.add_argument("--xml-short", dest="xml_short", default=False,
        type=bool,
        help="If the XML output should use short tag-names")

    parser.add_argument("--xml-omit-compartments",
        dest="xml_omit_compartments", type=str,
        help="Comma separated list of compartments to not report")

    parser.add_argument("--xml-omit-measures", dest="xml_omit_measures",
        type=str,
        help="Comma seperated list of measures to not report")

    parser.add_argument("--debug", dest="debug_level", default="warning",
        type=str, help="Set debugging level")

    args = parser.parse_args()

    #THE THREE SETTINGS DICTS
    grid_array_settings = {'animate': False}

    gridding_settings = {'use_otsu': True, 'median_coeff': 0.99,
            'manual_threshold': 0.05}

    grid_cell_settings = {'blob_detect': 'default'}

    #DEBUGGING
    LOGGING_LEVELS = {'critical': logging.CRITICAL,
                      'error': logging.ERROR,
                      'warning': logging.WARNING,
                      'info': logging.INFO,
                      'debug': logging.DEBUG}

    if args.debug_level in LOGGING_LEVELS.keys():

        logging_level = LOGGING_LEVELS[args.debug_level]

    else:

        logging_level = LOGGING_LEVELS['warning']

    logger = logging.getLogger('Scan-o-Matic Analysis')

    #XML
    xml_format = {'short': args.xml_short, 'omit_compartments': [],
                        'omit_measures': []}

    if args.xml_omit_compartments is not None:

        xml_format['omit_compartments'] = map(lambda x: x.strip(),
            args.xml_omit_compartments.split(","))

    if args.xml_omit_measures is not None:

        xml_format['omit_measures'] = map(lambda x: x.strip(),
            args.xml_omit_measures.split(","))

    logging.debug("XML-formatting is " + \
                "{0}, omitting compartments {1} and measures {2}.".format(
                ['long', 'short'][xml_format['short']],
                xml_format['omit_compartments'],
                xml_format['omit_measures']))

    #BLOB DETECTION
    args.b_detect = args.b_detect.lower()

    if args.b_detect not in ('default',):

        args.b_detect = 'default'

    grid_cell_settings['blob_detect'] = args.b_detect

    #GRID OTSU
    gridding_settings['use_otsu'] = args.g_otsu

    #ANIMATE THE WATCHED COLONY
    grid_array_settings['animate'] = args.animate

    #MATRICES
    if args.matrices is not None:

        pm = resource_analysis_support.get_pinning_matrices(args.matrices)
        logging.debug("Matrices: {0}".format(pm))

        if pm == [None] * len(pm):

            logging.error("No valid pinning matrices, aborting")
            parser.error("Check that you supplied a valid string...")
            

    else:

        pm = None

    #TIMES TO SAVE GRIDDING IMAGE
    if args.grid_times != None:

        try:

            grid_times = [int(grid_times)]

        except:

            try:

                grid_times = map(int, args.grid_times.split(","))

            except:

                logging.warning("ARGUMENTS, could not parse grid_times..." + \
                    " will only save the first grid placement.")

                grid_times = [-1]

    #INPUT FILE LOCATION
    if args.inputfile == None:

        parser.error("You need to specify input file!")

    in_path_list = args.inputfile.split(os.sep)

    try:

        fh = open(args.inputfile, 'r')

    except:

        parser.error('Cannot open input file, please check your path...')

    fh.close()

    #OUTPUT LOCATION
    output_path = ""

    if len(in_path_list) == 1:

        output_path = "."

    else:

        output_path = os.sep.join(in_path_list[:-1])

    if args.outputpath == None:

        output_path += os.sep + "analysis"

    else:

        output_path += os.sep + str(args.outputpath)

    #SPECIAL WATCH GRAPH
    if args.graph_watch != None:

        args.graph_watch = args.graph_watch.split(":")

        try:

            args.graph_watch = map(int, args.graph_watch)

        except:

            parser.error('The watched colony could not be resolved,' + \
                                ' make sure that you follow syntax')

        if len(args.graph_watch) != 3:

            parser.error('Bad specification of watched colony')

    #OUTPUT TO USER
    header_str = "The Project Analysis Script..."
    under_line = "-"

    print "\n\n{0}\n{1}\n\n".format(header_str.center(80),
            (len(header_str) * under_line).center(80))

    #LOGGER
    logger.setLevel(logging_level)
    logger.debug("Logger is ready!")

    #START ANALYSIS
    analysis.analyse_project(args.inputfile, output_path, pm, args.graph_watch,
        verbose=True, visual=False, manual_grid=args.manual_grid,
        grid_times=grid_times,
        xml_format=xml_format,
        suppress_analysis=args.suppress,
        grid_array_settings=grid_array_settings,
        gridding_settings=gridding_settings,
        grid_cell_settings=grid_cell_settings,
        logger=logger)
