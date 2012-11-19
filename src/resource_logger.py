#!/usr/bin/env python
"""Home-brewn logging tools"""
__author__ = "Martin Zackrisson, jetxee"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# CLASSES
#

class Log_Garbage_Collector(object):

    def __call__(self, *args):

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

    def _output(self, lvl, args):

        print "*{0}:\t{1}".format(lvl, args)

