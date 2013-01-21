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
import time
import collections
from itertools import chain
from subprocess import Popen, PIPE
import gobject
import threading

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


def get_pinnings_str(pinning_list):

    pinning_string = ""

    for p in pinning_list:

        if p is None:

            pinning_string += "None,"

        else:

            try:
                pinning_string += "{0}x{1},".format(*p)
            except:
                pinning_string += "None,"

    return pinning_string[:-1]

#
# CLASSES
#


class Experiment_Controller(controller_generic.Controller):

    def __init__(self, main_controller, logger=None):

        super(Experiment_Controller, self).__init__(main_controller,
            logger=logger)
        self._specific_controller = None

    def ask_destroy(self,*args, **kwargs):

        if self._specific_controller is not None:
            return self._specific_controller.ask_destroy(*args, **kwargs)
        else:
            return True

    def destroy(self):

        if self._specific_controller is not None:
            self._specific_controller.destroy()

    def _get_default_view(self):

        return view_experiment.Experiment_View(self, self._model)

    def _get_default_model(self):

        return model_experiment.get_gui_model()

    def get_mode(self):

        return self._experiment_mode

    def set_mode(self, widget, experiment_mode):

        view = self._view
        model = self._model
        self._experiment_mode = experiment_mode

        if experiment_mode == 'project':

            self._specific_controller = Project_Controller(
                self, model=model, view=view)

        elif experiment_mode == "gray":

            self._specific_controller = One_Controller(
                self, model=model, view=view)

        elif experiment_mode == 'color':

            self._specific_controller = One_Controller(
                self, model=model, view=view)

        else:
            self._experiment_mode = 'about'
            raise Bad_Stage_Call(experiment_mode)

class One_Controller(controller_generic.Controller):

    def __init__(self, parent, view=None, model=None,
        specific_model=None, logger=None):

        super(One_Controller, self).__init__(parent,
            view=view, model=model, logger=logger)

        if specific_model is not None:
            self._specific_model = specific_model
        else:
            self.build_new_specific_model()

        self._specific_model['experiments-root'] = \
            self.get_top_controller().paths.experiment_root

        view.set_controller(self)
        top = view_experiment.Top_One(self, model, 
            self._specific_model)
        stage = view_experiment.Stage_One(self, model,
            self._specific_model)
        view.set_top(top)
        view.set_stage(stage)
        stage.force_no_fixture()

    def build_new_specific_model(self):

        if self._parent.get_mode() == 'color': 
            sm_template = model_experiment.specific_one_color_model
        else:
            sm_template = model_experiment.specific_one_transparency_model

        sm = model_experiment.copy_model(sm_template)
        self._specific_model = sm
        return sm

    def destroy(self):

        sm = self._specific_model
        tc = self.get_top_controller()

        if sm['scanner'] is not None:
            tc.scanners.free(sm['scanner'], soft=True)

    def _set_project_id(self):

        sm = self._specific_model

        sm['experiment-id'] = \
            time.strftime("%d_%b_%Y__%H_%M_%S", time.gmtime())

        sm['experiment-root'] = os.sep.join((
            sm['experiments-root'], sm['experiment-id']))
 
        os.makedirs(sm['experiment-root'])

    def get_model_intro_key(self):

        sm = self._specific_model

        if sm['type'] == 'transparency':
            return 'one-stage-intro-transparency'
        elif sm['type'] == 'color':
            return 'one-stage-intro-color'

    def set_new_scanner(self, widget):

        sm = self._specific_model
        widget_model = widget.get_model()
        scanner = widget_model[widget.get_active()][0]

        scanners = self.get_top_controller().scanners

        if scanners.claim(scanner):

            #REMOVE PREVIOUS CLAIM
            if sm['scanner'] is not None:
                scanners.free(sm['scanner'])

            #UPDATE MODEL FOR CURRENT CLAIM
            sm['scanner'] = scanner

        else:

            self.get_view().get_stage().update_scanner()

        self._set_ready_to_run()

    def set_new_fixture(self, widget):

        sm = self._specific_model
        stage = self.get_view().get_stage()
        widget_model = widget.get_model()

        val = widget_model[widget.get_active()][0]
        if val == self._model['one-stage-no-fixture']:
            stage.set_progress('analysis', surpass=True)
            sm['fixture'] = False
        else:
            stage.set_progress('analysis', surpass=False)
            sm['fixture'] = val

        self._set_ready_to_run()

    def _set_ready_to_run(self):

        sm = self._specific_model
        stage = self.get_view().get_stage()

        if sm['fixture'] is not None and sm['scanner'] is not None:
            stage.set_run_stage('ready')
            
    def set_run(self, widget, run_command):

        run_command = run_command[0]
        stage = self.get_view().get_stage()
        if run_command == 'power-up':
            stage.set_run_stage('power-on')

        stage.set_run_stage('started')
        stage.set_run_stage('running')

        sm = self._specific_model
        sm['run'] = run_command

        scanners = self.get_top_controller().scanners
        scanner = scanners[sm['scanner']]

        if sm['stage'] is None:
            self._set_project_id()
            stage.set_progress('on')
            #POWER UP
            thread = threading.Thread(target=self._power_up, args=(scanner, stage))
            thread.start()

        elif run_command != 'complete':
            stage.set_progress('scan')
            thread = threading.Thread(self._scan, args=(scanner, stage))
            thread.start()
        else:
            stage.set_progress('off')    
            thread = threading.Thread(self._power_down, args=(scanner, stage))
            thread.start()
            
    def _power_up(self, scanner, stage):

        sm = self._specific_model
        if scanner.on():
            sm['stage'] = 'on'
            stage.set_progress('on', completed=True)
            if sm['run'] == 'power-up':
                stage.set_run_stage('ready')
            else:
                stage.set_progress('scan')
                self._scan(scanner, stage)

        else:
            scanner.free()
            stage.set_progress('on', failed=True)
            stage.set_progress('done', failed=True)

    def _scan(self, scanner, stage):

        sm = self._specific_model
        stage.set_run_stage('running')

        file_path = os.sep.join((sm['experiment-root'],
            'scan__{0}.tiff'.format(str(sm['image']).zfill(4))))

        scanner.scan(sm['scan-mode'], file_path, auto_off=False)

        stage.set_progress('scan', completed=True)

        if sm['run'] != 'scan':
            stage.set_progress('off')
            self._power_down(scanner, stage)
        else:
            sm['image'] += 1
            stage.set_run_stage('ready')

    def _power_down(self, scanner, stage):

        sm = self._specific_model
        stage.set_run_stage('running')
        if scanner.off():
            sm['stage'] = 'off'
            stage.set_progress('off', completed=True)
        else:
            stage.set_progress('off', failed=True)
        
        if sm['fixture'] != False:
            stage.set_progress('analysis')
            self._analysis(scanner, stage)
        else:
            scanner.free()
            sm['stage'] = 'done'
            stage.set_progress('done', completed=True)

    def _analysis(self, scaner, stage):

        scanner.free()
        stage.set_progress('done', completed=True)
        sm['stage'] = 'done'
        return False

class Project_Controller(controller_generic.Controller):

    #Input Bounds Validity
    bounds = {
        'duration': (14/60.0, 24*7),  # Hours
        'interval': (7, 3*60),  # Minutes
        'scans': (2, 1000)}

    def __init__(self, parent, view=None, model=None,
        specific_model=None, logger=None):

        super(Project_Controller, self).__init__(parent,
            view=view, model=model, logger=logger)

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

    def destroy(self):

        sm = self._specific_model
        tc = self.get_top_controller()

        if sm['scanner'] is not None:
            tc.scanners.free(sm['scanner'], soft=True)

    def set_project_root(self, widget):

        dir_list = view_experiment.select_dir(
            self._model['project-stage-select_root'],
            start_in=self._specific_model['experiments-root'])

        if dir_list is not None:

            self._specific_model['experiments-root'] = dir_list
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

        elif stage_call == "running":

            top = view_experiment.Top_Project_Running(self, m, sm)
            stage = view_experiment.Stage_Project_Running(self, m , sm)

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

            #REMOVE PREVIOUS CLAIM
            if sm['scanner'] is not None:
                scanners.free(sm['scanner'])

            #UPDATE MODEL FOR CURRENT CLAIM
            sm['scanner'] = scanner

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

    def start(self, *args):

        sm = self._specific_model
        tc = self.get_top_controller()
        scanner = tc.scanners[sm['scanner']]

        experiment_query = tc.config.get_default_experiment_query()
        experiment_query['-f'] = sm['fixture']
        experiment_query['-s'] = sm['scanner']
        experiment_query['-i'] = sm['interval']
        experiment_query['-n'] = sm['scans']

        if sm['experiments-root'] != '':
            experiment_query['-r'] = sm['experiments-root']

        experiment_query['-p'] = sm['experiment-prefix']
        experiment_query['-d'] = sm['experiment-desc']
        experiment_query['-c'] = sm['experiment-id']
        experiment_query['-u'] = scanner.get_uuid()

        experiment_query['-m'] = get_pinnings_str(sm['pinnings-list'])

        e_query_list = [tc.paths.experiment]
        e_query_list += list(chain.from_iterable(experiment_query.items()))
        e_query_list = map(str, e_query_list)

        stdin_path = tc.paths.experiment_stdin.format(
            tc.paths.get_scanner_path_name(scanner.get_name()))
        stdin = open(stdin_path, 'w')
        stdin.close()
        stdout_path = tc.paths.log_scanner_out.format(scanner.get_socket())
        stdout = open(stdout_path, 'w')
        stderr_path = tc.paths.log_scanner_err.format(scanner.get_socket())
        stderr = open(stderr_path, 'w')

        proc = Popen(e_query_list, stdout=stdout, stderr=stderr, shell=False)
        proc_type = 'scanner'

        self._logger.info(
            "Started experiment {0} ".format(sm['experiment-prefix']) + \
            "in directory {0} ".format(sm['experiments-root']) + \
            "on scanner {0}, fixture {1}".format(sm['scanner'], sm['fixture']))

        self._logger.debug("Command:\n" + " ".join(e_query_list))

        scanner.set_uuid()

        self.get_top_controller().add_subprocess(proc, proc_type, 
            stdin=stdin_path, stdout=stdout_path, stderr=stderr_path,
            pid=proc.pid, psm=sm, proc_name=sm['scanner'])

        self.set_view_stage(None, 'running')

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
