#!/usr/bin/env python
"""SubProcess Handlers for Analysis Processes and Experiment Processes."""
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
# INTERNAL DEPENDENCIES
#

from subproc_collection_interface import SubProc_Collection_Interface
import subproc_interface

#
# EXCEPTIONS
#


class WrongProcType(Exception):
    pass


class InvalidElementInterface(Exception):
    pass

#
# CLASSES
#


class _SubProc_Handler(SubProc_Collection_Interface):

    def __init__(self, proc_types):
        """Prototype SubProc Handler

        Implements all methods of the SubProc_Collection_Interface.

        :param proc_types: A list of SubProc types using the
        constants of the supproc_interface - module.
        """
        self._proc_type = proc_types
        self._store = set()
        self._count = 0

    def __iter__(self):
        """Returns an iter for all handled processes"""

        return self._store.__iter__()

    def count(self):
        """Returns precalculated size of handler's store"""

        return self._count

    def pop(self):
        """Returns element if element reports to be done.

        If nothing is completed, returns None.

        Note that it will only return one element at a time
        even if several are done. And note that the order is
        arbitrairy.
        """

        for elem in self:

            if elem.is_done():
                self._store.remove(elem)
                self._set_size()
                return elem

        return None

    def push(self, elem):
        """Adds subproc elem to handler.

        Push will fail if elem is already in handler.
        Push will fail if elem does not have the expected
        interface and the expected type-response

        :returns: Boolean if successful
        """

        if (self._verify_elem(elem) and not(self._store.issubset((elem,)))):
            self._store.add(elem)
            return True
        else:
            return False

    def _verify_elem(self, elem):
        """Throws exceptions if elem doesn't meet standard

        :returns: Boolean
        """
        ver = True
        if elem.get_type() not in self._proc_types:

            ver = False
            raise WrongProcType(elem)

        #FIXIT: TYPE NOT DONE YET
        if not(set(dir(subproc_interface.SubProc_Interface)).issubset(
                set(dir(elem)))):

            ver = False
            raise InvalidElementInterface(elem)

        return ver

    def _set_size(self):
        """Updates fast access size of handler"""

        self._count = len(self._store)


class Experiment_Handler(_SubProc_Handler):

    def __init__(self):

        super(Experiment_Handler, self).__init__(
            (subproc_interface.EXPERIMENT_SCANNING,
             subproc_interface.EXPERIMENT_REBUILD))


class Analysis_Handler(_SubProc_Handler):

    def __init__(self):

        super(Analysis_Handler, self).__init__(
            (subproc_interface.ANALYSIS,))
