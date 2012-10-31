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

#
# INTERNAL DEPENDENCIES
#

import src.model_experiment as model_experiment
import src.view_experiment as view_experiment
import src.controller_generic as controller_generic

#
# EXCEPTIONS
#

class Bad_Stage_Call(Exception): pass
class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class UnDocumented_Error(Exception): pass

#
# CLASSES
#

class Experiment_Controller(controller_generic.Controller):

    def __init__(self, window, main_controller):

        super(Experiment_Controller, self).__init__(window, main_controller)
        self._specific_controller = None

    def _get_default_view(self):

        return view_experiment.Experiment_View(self, self._model)

    def _get_default_model(self):

        return model_experiment.model

    def set_mode(self, widget, experiment_mode):

        view = self._view
        model = self._model

        if experiment_mode == 'project':

            self._specific_controller = Project_Controller(self._window,
                self, model=model, view=view)

        elif experiment_mode == "gray":

            err = Not_Yet_Implemented("Mode 'One Gray-Scale Scan'")
            raise err

        elif experiment_mode == 'color':

            err = Not_Yet_Implemented("Mode 'One Color Scan'")
            raise err

        else:

            raise Bad_Stage_Call(experiment_mode)


class Project_Controller(controller_generic.Controller):

    def __init__(self, window, parent, view=None, model=None,
        specific_model=None):

        super(Project_Controller, self).__init__(window, parent,
            view=view, model=model)

        #MODEL
        if specific_model is not None:
            self._specific_model = specific_model
        else:
            self.build_new_specific_model()

        #VIEW
        view.set_controller(self)
        self.set_view_stage(None, 'setup')

    def set_view_stage(self, widget, stage_call, *args, **kwargs):

        print widget, stage_call

    def build_new_specific_model(self):

        sm = model_experiment.copy_model(
            model_experiment.specific_project_model)
        self._specific_model = sm
        return sm
