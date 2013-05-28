#!/usr/bin/env python
"""Module runs subprocess that makes projects."""
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

import sys
import os
import logging
import threading
import time
from argparse import ArgumentParser
from ConfigParser import ConfigParser

#
# INTERNAL-DEPENDENCIES
#

import src.subproc.communicator as communicator

#
# CONSTANTS
#

LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG}

#
# CLASS
#


class Make_Project(object):

    META_PREFIX = "Prefix"
    META_ROOT = "Root"
    META_PINNING = "Pinning Matrices"

    CONFIG_IMAGES = "Image Paths"

    def __init__(self, inputFile, logger):

        self._time_init = time.time()
        self._running = None
        self._paused = False
        self._logger = logger
        self._set_from_file(inputFile)
        self._comm = communicator.Communicator(
            logger, self,  self._stdin, self._stdout, self._stderr)

        self._comm_thread = threading.Thread(target=self._comm.run)
        self._comm_thread.start()

    def _set_from_file(self, fpath):

        config = ConfigParser()
        config.readfp(open(fpath))

        tmpImgs = config.items(self.CONFIG_IMAGES)
        imgGen = ((int(k), v) for k, v in tmpImgs)
        self._image_paths = dict(imgGen)
        self._images_total = max(imgGen.keys()) + 1
        self._meta_data = {}
        self._outfile_path = ""

    def run(self):

        self._time_run_start = time.time()
        self._running = True
        self._image_i = 0

        #WRITE METADATA HEADER
        pass

        #DO THE IMAGES
        while self._running:

            if self._image_i in self._image_paths:

                #DO THE FIRST PASS
                pass

            self._image_i += 1

            while self._paused and self._running:

                time.sleep(0.42)

        #Clean-up
        self._running = False
        self._comm.set_terminate()
        self._comm_thread.join()

    def get_current_step(self):

        return self._image_i + 1

    def get_total_iterations(self):

        return self._images_total

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

        return ("__PREFIX__ {0}".format(self._meta_data[self.META_PREFIX]),
                "__SCANNER__ <NONE>",
                "__ROOT__ {0}\n".format(self._meta_data[self.META_ROOT]),
                "__1-PASS FILE__ {0}".format(self._outfile_path),
                "__PINNINGS__ {0}".format(self._meta_data[self.META_PINNING]))

#
# RUN BEHAVIOR
#

if __name__ == "__main__":

    print "MAKE PROJECT WAS CALLED WITH:"
    print " ".join(sys.argv)
    print
    print

    parser = ArgumentParser(
        description="""The Make Project script compiles a new project
        as if it was run for the first time, but uses images allready
        existing.""")

    parser.add_argument("-i", "--input-file", type=str, dest="inputfile",
                        help="Settings File for the compilation",
                        metavar="PATH")

    parser.add_argument("-l", "--logging", type=str, dest="logging",
                        help="Logging level {0}".format(
                            LOGGING_LEVELS.keys()),
                        metavar="LOGGING LEVEL")

    args = parser.parse_args()

    if args.inputfile is None or not os.path.isfile(args.inputfile):

        parser.error("Could not find file {0}".format(args.inputfile))

    #LOGGING
    if args.logging in LOGGING_LEVELS.keys():

        args.logging = LOGGING_LEVELS[args.debug_level]

    else:

        args.logging = LOGGING_LEVELS['warning']

    logger = logging.getLogger('Scan-o-Matic Make Project')
    logger.setLevel(args.logging)
    logger.debug("Logger is ready!")

    #Making the project
    mp = Make_Project(args.inputfile, logger)
    mp.run()
