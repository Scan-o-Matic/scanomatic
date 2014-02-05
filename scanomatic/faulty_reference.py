"""FaultyReference is a debug class to find how outdated thing are called"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger

#
# CLASSES
#


class FaultyReference(object):

    def __init__(self, name, base=None):

        name = base is None and name or self.NAME_JOINER.join((base, name))

        self._faultyLogger = logger.Logger(name)
        self._faultyName = name
        self._faultyLogger.critical(
            "I've been requested though I have no right to exist")

    def __getattr__(self, name):

        if (name == "NAME_JOINER"):
            return "."
        elif (name in ("_faultyLogger", "_faultyName")):
            return self.__dict__[name]

        return FaultyReference(self.NAME_JOINER.join((self._faultyName, name)))

    def __setattr__(self, name, value):

        if (name in ("_faultyLogger", "_faultyName")):
            self.__dict__[name] = value
            return

        self._faultyLogger.critical("Tried to set property '{0}' with '{1}'".format(
            name, value) +
            ", this is impossible since reference no longer in use")

    def __delattr__(self, name):

        self._faultyLogger.critical(
            "Tried to delete property '{0}'. Impossible".format(name))

    def __call__(self, *args, **kwargs):

        self._faultyLogger.critical(
            "Tried to call me with '{0}' and' {1}'. I don't exist!".format(
                args, kwargs))
