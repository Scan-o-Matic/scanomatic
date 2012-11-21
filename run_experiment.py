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

import logging
import threading
import os
import sys
import time
import uuid
from argparse import ArgumentParser

#
# SCANNOMATIC LIBRARIES
#

import src.resource_scanner as resource_scanner
import src.resource_first_pass_analysis as resource_first_pass_analysis
import src.resource_fixture as resource_fixture
import src.resource_path as resource_path
import src.resource_logger as resource_logger
import src.resource_app_config as resource_app_config

#
# EXCEPTIONS
#

class Not_Initialized(Exception): pass

#
# CLASSES
#

class Experiment(object):

    def __init__(self, run_args=None, logger=None, **kwargs):

        self.paths = resource_path.Paths(program_path)
        self.fixtures = resource_fixture.Fixtures(self.paths)
        self.config = resource_app_config.Config(self.paths)
        self.scanners = resource_scanner.Scanners(self.paths, self.config)

        self._running = True
        self._stdin_pipe_deamon = threading.Thread(target=self._stdin_deamon, args=self)
        self._stdin_pipe_deamon.start()

        self._scan_threads = list()

        self.set_logger(logger)

        self._scanned = 0

        self._initialized = False
        
        if run_args is not None:
            self._set_settings_from_run_args(run_args)
        elif kwargs is not None:
            self._set_settings_from_kwargs(kwargs)

    def _stdin_deamon(self):

        while self._running:
            line = sys.stdin.readline().strip()
            sys.stdin.flush()
            if line == "__QUIT__":
                self._running = False
                #REPORT THAT GOT QUIT REQUEST

    def _generate_uuid(self):

        self._uuid = uuid.uuid1().get_urn().split(":")[-1]

    def _set_settings_from_kwargs(self):

        raise Not_Initialized("Setting from kwarg not done yet")

    def _set_settings_from_run_args(self, run_args):

        self._interval = run_args.interval
        self._max_scans = run_arsg.number_of_scans

        self._scanner = self.scanners[run_args.scanner]
        self._fixture = self.fixtures[run_args.fixture]

        self._root = run_args.root
        self._prefix = run_args.prefix
        self._im_filename_pattern = os.sep.join(self._root, self._prefix,
            self.paths.scan_image_name_pattern)

        self._description = run_args.description
        self._code = run_args.code

        self._uuid = run_args.uuid
        if self._uuid is None or self._uuid == "":
            self._generate_uuid()

        self._initialized = True

    def set_logger(self, logger):

        if logger is None:
            self._logger = resource_logger.Fallback_Logger()
        else:
            self._logger = logger

    def run(self):

        if not self._initialized:
            raise Not_Initialized()

        self._logger.info("Entering main loop")
        timer = time.time()

        while self._running:

            if (time.time() - timer)*60 > self._interval:

                timer = time.time()
                self._get_image()

            time.sleep(0.1)

        self._join_threads()

    def _get_image(self):

        if self._running:

            self._logger.info("Aquiring image {0}".format(self._scanned))

            #THREAD IMAGE AQ AND ANALYSIS
            thread = threading.Thread(target=self._scan_and_analyse, args=self._scanned)

            thread.start()


            #CHECK IF ALL IS DONE
            self._scanned += 1

            if self._scanned > self._max_scans:

                self._logger.info("That was the last image")
                self._running = False

                return False

        else:

            self._logger.info("Aborting, no image aquired")
            return False

    def _scan_and_analyse(self, im_index):

        #SCAN
        self._scanner.scan(filename=self._im_filename_pattern.format(im_index))

        #FREE SCANNER IF LAST
        if not self._running or self._scanned => self._max_scans:
            self._scanner.free()

        #ANALYSE
        im_dict = resource_first_pass_analysis.analyse(
            file_name=self._im_filename_pattern.format(im_index),
            fixture=self._fixture)

        #APPEND TO FILE
        self._write_im_row(im_index, im_dict)

    def _write_header_row(self):

        pass

    def _write_im_row(self, im_index, im_dict):

        pass

    def _join_threads(self):

        self._logger.info("Waiting for all scans and analysis to finnish")

        while threading.active_count() > 0:
            time.sleep(0.5)

        self._scanner.free()

        self._logger.info("All threads are finnished, experiment run is done")

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

    parser.add_argument('m', '--matrices', type=str, dest='pinning',
        help='List of pinning matrices')

    parser.add_argument('-r', '--root', type=str, dest='root',
        help='Projects root')

    parser.add_argument('-p', '--prefix', type=str, dest='prefix',
        help='Project prefix')

    parser.add_argument('-d', '--description', type=str, dest='description',
        help='Project description')

    parser.add_argument('-c', '--code', type=str, dest='code',
        help='Identification code for the project as supplied by the planner')

    parser.add_argument('-u', '--uuid', type=str, dest='uuid',
        help='UUID to indentify self with scanner reservation')

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

    if args.pinning is not None:
        args.pinning = get_pinnings_list(args.pinning)

    if args.pinning is None:
        parser.error("Bad pinning supplied")        

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

    e = Experiment(run_args = args, logger=logger)
    e.run()
