#!/usr/bin/env python
"""Script that runs the image aquisition"""

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

import gobject
import os
from argparse import ArgumentParser

#
# SCANNOMATIC LIBRARIES
#

import src.resource_power_manager as resource_power_manager
import src.resource_fixture_image as resource_fixture_settings_image


def run(args, logging_level):

    pass

#
# COMMAND LINE BEHAVIOUR
#

if __name__ == "__main__":

    parser = ArgumentParser(description="""Runs a session of image gathering
given certain parameters and creates a first pass analysis file which is the
input file for the analysis script.""")

    parser.add_argument('-f', '--fixture', type=str, dest="fixture",
        help='Path to fixture config file')

    parser.add_argument('-s', '--scanner', type=str, dest='scanner',
        help='Scanner to be used')

    parser.add_argument('-i', '--interval', type=float, default=20.0,
        dest='interval',
        help='Minutes between scans')

    parser.add_argument('-n', '--number-of-scans', type=int, default=217,
        dest='number_of_scans',
        help='Number of scans requested')

    parser.add_argument('-r', '--root', type=str, dest='root',
        help='Projects root')

    parser.add_argument('-p', '--prefix', type=str, dest='prefix',
        help='Project prefix')

    parser.add_argument('-d', '--description', type=str, dest='description',
        help='Project description')

    parser.add_argument('-c', '--code', type=str, dest='code',
        help='Identification code for the project as supplied by the planner')

    parser.add_argument("--debug", dest="debug", default="warning",
        type=str, help="Sets debugging level")

    args = parser.parse_args()

    #DEBUGGING
    LOGGING_LEVELS = {'critical': logging.CRITICAL,
                      'error': logging.ERROR,
                      'warning': logging.WARNING,
                      'info': logging.INFO,
                      'debug': logging.DEBUG}


    #TESTING FIXTURE FILE
    try:
        fs = open(args.fixture, 'r')
        fs.close()
    except:
        parser.error("Can't find any file at '{0}'".format(args.fixture))


    #SCANNER
    if args.scanner is None:
        parser.error("Without specifying a scanner, this makes little sense")
    #elif get_scanner_resource(args.scanner) is None

    #INTERVAl
    if 7 > args.interval > 4*60:
        parser.error("Interval is out of allowed bounds!")

    #NUMBER OF SCANS
    if 2 > args.number_of_scans > 1000:
        parser.error("Number of scans is out of bounds")

    #EXPERIMENTS ROOT
    if args.root is None or os.path.isdir(args.root) == "False":
        parser.error("Experiments root is not a directory")

    #PREFIX
    if args.prefix is None or \
        os.path.isdir(os.sep.join(args.root, args.prefix)):

        parser.error("Prefix is a duplicate or invalid")

    #CODE
    if args.code is None or args.code == "":

        pass
        #args.code = uuid.

    #LOGGING
    if args.debug in LOGGING_LEVELS.keys():

        logging_level = LOGGING_LEVELS[args.debug]

    else:

        logging_level = LOGGING_LEVELS['warning']

    logger = logging.getLogger('Scan-o-Matic First Pass Analysis')

    log_formatter = logging.Formatter('\n\n%(asctime)s %(levelname)s:' + \
                    ' %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S\n')

    #NEED NICER PATH THING
    hdlr = logging.FileHandler(outdata_files_path + "analysis.run", mode='w')
    hdlr.setFormatter(log_formatter)
    logger.addHandler(hdlr)

    logger.setLevel(logging_level)

    logger.debug("Logger is ready! Arguments are ready! Lets roll!")
