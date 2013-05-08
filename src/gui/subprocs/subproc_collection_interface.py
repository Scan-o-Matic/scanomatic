#!/usr/bin/env python
"""SubProcess Collections Interface"""
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


class InvalidSubProcessCollection(Exception):

    pass

#
# CLASSES
#


class SubProc_Collection_Interface(object):

    def __init__(self, *args, **kwargs):
        """Common interface for SubProc Collection objects.

        The super functions should not be run, instead they
        work by guaranteeing that the subclass interface has
        implemented all neccesary methods overriding the
        original ones as all these will throw exceptions.

        Iterator
        ========

        An iterator that iterates over the collection's
        members.

        Count
        =====

        Returns size of collection

        Pop
        ===

        Pop will return an element or None

        Push
        ====

        Appends an element to Collection
        Returns boolean
        """

        raise InvalidSubProcessCollection(
            "{0} not fully implemented".format(self))

    def __iter__(self):

        raise InvalidSubProcessCollection(
            "{0} not fully implemented".format(self))

    def count(self):

        raise InvalidSubProcessCollection(
            "{0} not fully implemented".format(self))

    def push(self, elem):

        raise InvalidSubProcessCollection(
            "{0} not fully implemented".format(self))

    def pop(self):

        raise InvalidSubProcessCollection(
            "{0} not fully implemented".format(self))
