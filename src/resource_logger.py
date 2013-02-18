#!/usr/bin/env python
"""Home-brewn logging tools"""
__author__ = "Martin Zackrisson, jetxee"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENICES
#

import logging
import traceback
import sys

#
# METHODS
#


def print_more_exc():

    tb = sys.exc_info()[2]  # Get the traceback object

    while tb.tb_next:
        tb = tb.tb_next

    stack = []

    f = tb.tb_frame
    while f:
        stack.append(f)
        f = f.f_back

    stack.reverse()

    #traceback.print_exc()
    for frame in stack[:-3]:
        print "\n\tFrame {0} in {1} at line {2}".format(frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno)

    print "\n\n"


#
# CLASSES
#

class Logging_Log(object):

    def __init__(self, logging_logger):

        self._logger = logging_logger

    def __call__(self, *args, **kwargs):

        self._logger.info("\t".join((str(args), str(kwargs))))

    def warning(self, *args, **kwargs):

        self._logger.warning(*args, **kwargs)

    def info(self, *args, **kwargs):

        self._logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):

        self._logger.debug(*args, **kwargs)

    def error(self, *args, **kwargs):

        self._logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):

        self._logger.critical(*args, **kwargs)

    def exception(self, *args, **kwargs):

        self._logger.exception(*args, **kwargs)
                
    
class Log_Garbage_Collector(object):

    def __call__(self, *args, **kwargs):

        pass

    def warning(self, *args, **kwargs):

        pass

    def info(self, *args, **kwargs):

        pass

    def debug(self, *args, **kwargs):

        pass

    def error(self, *args, **kwargs):

        pass

    def critical(self, *args, **kwargs):

        pass

    def exception(self, *args, **kwargs):

        pass

class Fallback_Logger(object):

    def __call__(self, *args, **kwargs):

        args += zip(kwargs.items())

        if len(args) >= 2:
            self._output(args[0], args[1:])
        else:
            self._output(args, "")

    def warning(self, *args):

        self._output("WARNING", args)

    def debug(self, *args):

        self._output("DEBUG", args)

    def info(self, *args):

        self._output("INFO", args)

    def error(self, *args):

        self._output("ERROR", args)

    def critical(self, *args):

        self._output("CRITICAL", args)

    def exception(self, *args):

        self._output("INVOKED EXCEPTION", args)
        try:
            x = 1/0
        except Exception, err:
            print_more_exc()

    def _output(self, lvl, args):

        print "*{0}:\t{1}".format(lvl, args)
