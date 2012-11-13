#!/usr/bin/env python
"""The Experiment Controller"""
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

#
# INTERNAL DEPENDENCIES
#

import src.controller_generic as controller_generic
import src.view_subprocs as view_subprocs
import src.model_subprocs as model_subprocs

#
# EXCEPTIONS
#

class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class UnDocumented_Error(Exception): pass

#
# FUNCTIONS
#

#
# CLASSES
#

class Subprocs_Controller(controller_generic.Controller):

    def __init__(self, window, main_controller):

        super(Subprocs_Controller, self).__init__(window, main_controller,
            specific_model=model_subprocs.get_composite_specific_model())

        gobject.timeout_add(1000, self._subprocess_callback)

    def _get_default_view(self):

        return view_subprocs.Subprocs_View(self, self._model, self._specific_model)

    def _get_default_model(self):

        return model_subprocs.get_gui_model()

    def add_subprocess(self, proc, stdin=None, stdout=None, stderr=None,
                        pid=None, proc_name=None):

        self._subprocesses.append((proc, pid, stdin, stdout, stderr, proc_name))

    def get_subprocesses(self, by_name=None):
        """INCOMPLETE"""
        ret = [p for p in self._specific_model['scanner-procs'] 
                if (by_name is not None and p[-1] == by_name or True)]

        return ret

    def _subprocess_callback(self):

        self._view.update()

        return True


    def produce_running_scanners(self, widget):

        pass

    def produce_running_analysis(self, widget):

        pass
