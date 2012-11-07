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

import re
import os
import collections

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
# FUNCTIONS
#

def zero_generator():
    return 0

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

        sm = self._specific_model
        m = self._model
        view = self.get_view()

        if stage_call == "setup":

            top = view_experiment.Top_Project_Setup(self, m , sm)
            stage = view_experiment.Stage_Project_Setup(self, m , sm)

        else:

            err = "{0} called for {1}".format(widget, stage_call)
            raise Bad_Stage_Call(err)

        view.set_top(top)
        view.set_stage(stage)

    def build_new_specific_model(self):

        sm = model_experiment.copy_model(
            model_experiment.specific_project_model)
        self._specific_model = sm
        return sm

    def check_prefix_dupe(self, widget):

        stage = self._view.get_stage()
        sm = self._specific_model
        t = widget.get_text()

        if re.match("^[A-Za-z0-9_-]*$", t) and t != "":

            if os.path.isdir(sm['experiments-root'] + os.sep + t):

                stage.set_prefix_status(False)

            else:

                stage.set_prefix_status(True)

        else:

            stage.set_prefix_status(False)


    def check_experiment_duration(self, widget, widget_name):

        sm = self._specific_model
        stage = self._view.get_stage()

        #Input Bounds Validity
        bounds = {
            'duration': (14, 24*7),  # Hours
            'interval': (7, 3*60),  # Minutes
            'scans': (2, 1000)}

        #Parsing input
        str_val = widget.get_text()
        val = None

        if widget_name == 'interval':

            try:
                val = float(re.findall(r'^ ?([0-9.]*)', str_val)[0])
            except:
                pass

        elif widget_name == 'scans':

            try:
                val = int(re.findall(r'^ ?([0-9]*)', str_val)[0])                
            except:
                pass

        elif widget_name == 'duration':

            str_val = str_val.lower()
            duration_dict = collections.defaultdict(zero_generator)
            try:
                hits = re.findall(r'([0-9]{1,2}) ?(days?|hours?|mins?)', str_val)
                for hit_val, hit_type in hits:
                    duration_dict[hit_type[0]] = int(hit_val)

                val = duration_dict['d'] * 24 + duration_dict['h'] + duration_dict['m'] / 60.0
            except:
                pass

            if val == 0:
                val = None

        if val is not None:

            #Checking bounds
            val_is_adjusted = False
            if bounds[widget_name][0] > val:
                val = bounds[widget_name][0]
                val_is_adjusted = True
            elif val < bounds[widget_name][1]:
                val = bounds[widget_name][1]
                val_is_adjusted = True

            if val_is_adjusted:
                stage.set_duration_warning(widget_name)    
            else:
                stage.remove_duration_warning(widget_name)

            sm[widget_name] = val
             
            #Update order of entry made
            dso = sm['duration-settings-order']
            dso.pop(dso.index(widget_name))
            dso.append(widget_name)

        else:

            stage.set_duration_warning(widget_name)    

        print widget_name, val
