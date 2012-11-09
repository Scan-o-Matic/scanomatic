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

        return model_experiment.get_gui_model()

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

    #Input Bounds Validity
    bounds = {
        'duration': (14/60.0, 24*7),  # Hours
        'interval': (7, 3*60),  # Minutes
        'scans': (2, 1000)}

    def __init__(self, window, parent, view=None, model=None,
        specific_model=None):

        super(Project_Controller, self).__init__(window, parent,
            view=view, model=model)

        #MODEL
        if specific_model is not None:
            self._specific_model = specific_model
        else:
            self.build_new_specific_model()
        self._specific_model['experiments-root'] = \
            self.get_top_controller().paths.experiment_root

        #VIEW
        view.set_controller(self)
        self.set_view_stage(None, 'setup')

    def set_project_root(self, widget):

        dir_list = view_experiment.select_dir(
            self._model['project-stage-select_root'])

        if dir_list is not None:

            self._specific_model['experiments-root'] = dir_list[0]
            self.check_prefix_dupe(widget=None)
            self._view.get_stage().update_experiment_root()

    def set_project_id(self, widget, event):

        self._specific_model['experiment-id'] = widget.get_text()

    def set_project_description(self, widget, event):

        self._specific_model['experiment-desc'] = widget.get_text()

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

    def check_prefix_dupe(self, widget=None):

        stage = self._view.get_stage()
        sm = self._specific_model
        if widget is None:
            t = sm['experiment-prefix']
        else:
            t = widget.get_text()

        if t is not None and re.match("^[A-Za-z0-9_-]*$", t) and t != "":

            if os.path.isdir(sm['experiments-root'] + os.sep + t):

                stage.set_prefix_status(False)
                sm['experiment-prefix'] = None
            else:

                stage.set_prefix_status(True)
                sm['experiment-prefix'] = t
        else:

            stage.set_prefix_status(False)

        self.set_allow_run()

    def check_experiment_duration(self, widget, widget_name):

        sm = self._specific_model
        stage = self._view.get_stage()


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
            val_is_adjusted = self._set_duration_in_model(widget_name, val)
            if val_is_adjusted:
                stage.set_duration_warning(widget_name)    
            else:
                stage.remove_duration_warning(widget_name)

            #Update order of entry made
            dso = sm['duration-settings-order']
            dso.pop(dso.index(widget_name))
            dso.append(widget_name)

            self._check_duration_consistencies()
            stage.set_other_duration_values()

        else:

            stage.set_duration_warning(widget_name)    


    def _check_duration_consistencies(self):

        sm = self._specific_model
        dso = sm['duration-settings-order']
        inconsistent = True

        for pos in range(len(dso)):

            if dso[pos] == 'interval':

                t = sm['duration'] * 60.0 / (sm['scans'] - 1)
                inconsistent = self._set_duration_in_model('interval', t)

            elif dso[pos] == 'duration':

                t = sm['interval'] * (sm['scans'] - 1) / 60.0
                inconsistent = self._set_duration_in_model('duration', t)
            
            else:

                t = int(sm['duration'] * 60 / sm['interval']) + 1
                inconsistent = self._set_duration_in_model('scans', t)

            if inconsistent == False:

                break

    def _set_duration_in_model(self, duration_name, val):

        sm = self._specific_model
        got_adjusted = True

        if val != sm[duration_name]:

            if val <= self.bounds[duration_name][1] and \
                val >= self.bounds[duration_name][0]:

                got_adjusted = False
                sm[duration_name] = val

            elif val < self.bounds[duration_name][0]:
                sm[duration_name] = self.bounds[duration_name][0]
            else:
                sm[duration_name] = self.bounds[duration_name][1]

        return got_adjusted

    def set_new_scanner(self, widget):

        row = widget.get_active()
        w_model = widget.get_model()
        sm = self._specific_model

        scanners = self.get_top_controller().scanners

        scanner = w_model[row][0]

        if scanners.claim(scanner):
            if sm['scanner'] is not None:
                scanners.free(sm['scanner'])

            sm['scanners'] = scanners

        else:

            self._view.get_stage().warn_scanner_claim_fail()

    def set_new_fixture(self, widget):

        row = widget.get_active()
        model = widget.get_model()

        fixtures = self.get_top_controller().fixtures

        fixtures[model[row][0]].set_experiment_model(self._specific_model, 
            default_pinning = self._model['pinning-default'])
        self._view.get_stage().set_pinning()
        self.set_allow_run()

    def set_pinning(self, widget, plate):

        plate_i = plate - 1
        model = widget.get_model()
        row = widget.get_active()
        key = model[row][0]
        pm = self._model['pinning-matrices'][key]
        self._specific_model['pinnings-list'][plate_i] = pm
        self.set_allow_run()

    def start(self):

        print "!!START"
        print  self._specific_model

    def set_allow_run(self):

        self._view.get_top().set_allow_next(self.get_ready_to_run())

    def get_ready_to_run(self):

        sm = self._specific_model
        is_ok = True

        if sm['fixture'] is None or sm['scanner'] is None:

            is_ok = False

        if sm['experiment-prefix'] is None:

            is_ok = False

        if sm['pinnings-list'] is None or \
            sum([p is None for p in sm['pinnings-list']]) == \
            len(sm['pinnings-list']):

            is_ok = False

        if sm['marker-path'] is None or sm['grayscale'] == False or \
            sm['grayscale-area'] is None or ['plate-areas'] == 0 or \
            sm['plate-areas'] is None:

            is_ok = False

        return is_ok
