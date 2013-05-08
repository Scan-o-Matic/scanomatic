#!/usr/bin/env python
"""SubProcess Interface"""
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

#
# EXCEPTIONS
#


class InvalidSubProc(Exception):
    pass

#
# CONSTANTS
#

EXPERIMENT_SCANNING = 11
EXPERIMENT_REBUILD = 12
ANALYSIS = 20

#
# CLASSES
#


class SubProc_Interface(object):

    def __init__(self, proc_type):

        """Common interface for SubProc Collection objects.

        The super functions should not be run, instead they
        work by guaranteeing that the subclass interface has
        implemented all neccesary methods overriding the
        original ones as all these will throw exceptions.

        This interface details the intent of each of those
        required methods.

        Init
        ====

        Initiation should set the process type according
        using one of the following constants in this
        module:

        EXPERIMENT_SCANNING
        -------------------

        The processes running scans

        EXPERIMENT_REBUILD
        ------------------

        The processes involved in rebuilding projects from
        already scanned images

        ANALYSIS
        --------

        Processes running analysis on projects

        get_type
        ========

        The SubProc_Interface.get_type() should the report
        the proc_type of the instance as described in the init

        is_done
        =======

        Verifies that the subprocess is still alive and returns
        a boolean

        is_paused
        =========

        Verifies if the subprocess is paused

        get_parameters
        ==============

        Returns the parameters that was used to initiate the subprocess

        get_progress
        ============

        The progress in precent

        get_current
        ===========

        The current iteration step

        get_total
        =========

        The total iterations the subprocess intends to go through, should
        be an int.

        set_pause
        =========

        Cuases the subprocess to pause
        Returns boolean indicating success

        set_terminate
        =============

        Causes the subprocess to terminate prematurely
        Returns boolean indicating success

        set_unpause
        ===========

        Causes the subprocess to resume
        Returns boolean indicating success
        """

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def get_type(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def is_alive(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def is_paused(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def is_done(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def get_parameters(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def set_pause(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def set_terminate(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))

    def set_unpause(self):

        raise InvalidSubProc(
            "{0} not fully implemented".format(self))
