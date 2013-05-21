#!/usr/bin/env python
"""The Experiment Controller"""
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

import gobject
import threading

#
# INTERNAL DEPENDENCIES
#

import src.controller_generic as controller_generic
import src.view_subprocs as view_subprocs
import src.model_subprocs as model_subprocs
#import src.model_experiment as model_experiment
from src.gui.subprocs.analysis_queue import Analysis_Queue
import src.gui.subprocs.gui_subprocesses as gui_subprocesses
import src.gui.subprocs.process_handler as process_handler
import src.gui.subprocs.subproc_interface as subproc_interface
import src.gui.subprocs.progress_responses as progress_responses
import src.gui.subprocs.reconnect as reconnect
import src.gui.subprocs.live_projects as live_projects

#
# EXCEPTIONS
#


class No_View_Loaded(Exception):
    pass


class Not_Yet_Implemented(Exception):
    pass


class UnDocumented_Error(Exception):
    pass

#
# FUNCTIONS
#

#
# CLASSES
#


class Subprocs_Controller(controller_generic.Controller,
                          progress_responses.Progress_Responses):

    ANALYSIS = "A"
    EXPERIMENT_SCANNING = "ES"
    EXPERIMENT_REBUILD = "ER"

    def __init__(self, main_controller, logger=None):

        super(Subprocs_Controller, self).__init__(
            main_controller,
            specific_model=model_subprocs.get_composite_specific_model(),
            logger=logger)

        self._tc = self.get_top_controller()
        self._project_progress = live_projects.Live_Projects(
            main_controller.paths, self._model)

        revive_processes = reconnect.Reconnect_Subprocs(self, logger)
        thread = threading.Thread(target=revive_processes.run)
        thread.start()
        gobject.timeout_add(131, self._revive_progress_callback, thread)

        #The Analysis_Queue makes program (more) well behaved
        #in terms of resource usage
        self._queue = Analysis_Queue()

        #Initiating the subprocess handlers
        self._experiments = process_handler.Experiment_Handler()
        self._analysises = process_handler.Analysis_Handler()

        #Updating model
        self._specific_model['queue'] = self._queue
        self._specific_model['experiments'] = self._experiments
        self._specific_model['analysises'] = self._analysises

    def _revive_progress_callback(self, thread):

        if thread.is_alive() is False:
            self._subprocess_callback()
            gobject.timeout_add(6421, self._subprocess_callback)
            return False
        else:
            return True

    def _get_default_view(self):

        return view_subprocs.Subprocs_View(self, self._model, self._specific_model)

    def _get_default_model(self):

        tc = self.get_top_controller()
        return model_subprocs.get_gui_model(paths=tc.paths)

    def ask_destroy(self):
        """This is to allow the fake destruction always"""
        return True

    def destroy(self):
        """Subproc is never destroyed, but its views always allow destruction"""
        pass

    def set_project_progress(self, prefix, stage, value, experiment_dir=None,
                             first_pass_file=None, analysis_path=None):

        return self._project_progress.set_status(
            prefix, stage, value,
            experiment_dir=experiment_dir,
            first_pass_file=first_pass_file,
            analysis_path=analysis_path)

    def remove_live_project(self, prefix):

        self._project_progress.remove_project(prefix)

    def get_remaining_scans(self):

        img_tot = 0
        img_done = 0

        for proc in self._experiments:

            img_tot += proc.get_total()
            img_done += proc.get_current()

        return img_tot - img_done

    def add_subprocess_directly(self, ptype, proc):
        """Adds a proc, should be used only by reconnecter"""

        if (ptype == self.EXPERIMENT_SCANNING or
                ptype == self.EXPERIMENT_REBUILD):

            success = self._experiments.push(proc)
            proc.set_callback_parameters(self._set_experiment_started)

        elif ptype == self.ANALYSIS:

            success == self._analysises.push(proc)

        return success

    def add_subprocess(self, ptype, **params):
        """Adds a new subprocess.

        If it can be started, it will be immidiately, else
        if will be queued.

        **params should include sufficient information for the
        relevant gui_subprocess-class to launch
        """

        if ptype == self.EXPERIMENT_SCANNING:

            success = self.add_subprocess_directly(
                ptype,
                gui_subprocesses.Experiment_Scanning(self._tc, **params))

        elif ptype == self.ANALYSIS:

            success = self._queue.push(params)

        elif ptype == self.EXPERIMENT_REBUILD:

            success = self.add_subprocess_directly(
                ptype,
                gui_subprocesses.Experiment_Rebuild(self._tc, **params))

        self._logger.info("{0} {1} with parameters {2}".format(
            ["Failed to add", "Added"][success],
            ptype, params))

    def _make_analysis_from_experiment(self, param):
        """Queries and experiment process to run defualt analysis

        The analysis is placed in the queue and run with default
        parameters, only adjusting for the path to the specific
        project.

        :param proc: A finished experiment process.
        :return boolean: Success statement
        """

        if param is not None and 'experiments-root' in param:

            self.add_subprocess(self.ANALYSIS,
                                experiments_root=param['experiments_root'],
                                experiment_prefix=param['prefix'])

            return True

        return False

    def _set_experiment_started(self, param):

            self.set_project_progress(
                param['prefix'], 'EXPERIMENT', 'RUNNING',
                experiment_dir=param['experiments-root'],
                first_pass_file=param['1-pass file'])

    def _set_experiment_completed(self, prefix):

            self.set_project_progress(prefix, 'EXPERIMENT', 'COMPLETED')
            self.set_project_progress(prefix, 'ANALYSIS', 'AUTOMATIC')

    def _set_analysis_started(self, param):

            self.set_project_progress(
                param['prefix'], 'ANALYSIS', 'RUNNING',
                experiment_dir=param['experiments-root'],
                first_pass_file=param['1-pass file'],
                analysis_path=param['analysis-dir'])

    def _experiment_is_alive(self, is_alive, experiment):

        if is_alive is False:

            #Place analysis in queue
            experiment.set_callback_parameters(
                self._make_analysis_from_experiment)

            #Update live project status
            experiment.set_callback_prefix(self._set_experiment_completed)

            experiment.close_communications()
            self._experiments.remove(experiment)

    def _set_analysis_completed(self, prefix):

        self.set_project_progress(
            prefix, 'ANALYSIS', 'COMPLETED')
        self.set_project_progress(
            prefix, 'UPLOAD', 'LAUNCH')

    def _set_analysis_failed(self, prefix):

        self.set_project_progress(
            prefix, 'ANALYSIS', 'FAILED')

    def _analysis_is_alive(self, is_alive, analysis):

        if is_alive is False:

            if analysis.get_exit_code() == 0:

                analysis.set_callback_prefix(self._set_analysis_completed)
            else:

                analysis.set_callback_prefix(self._set_analysis_failed)

            analysis.close_communications()
            self._analysises.remove(analysis)

    def _subprocess_callback(self):
        """Callback that checks on finished stuff etc"""

        sm = self._specific_model
        tc = self._tc

        #
        #   1 CHECKING EXPERIMENTS IF ANY IS DONE
        #

        for experiment in self._experiments:

            experiment.set_callback_is_alive(
                self._experiment_is_alive)

            experiment.update()

        #
        #   2. CHECKING IF ANY NEW ANALYSIS MAY BE STARTED
        #

        new_analsysis = self._queue.pop()

        if new_analsysis is not None:

            if 'comm_id' not in new_analsysis:
                new_analsysis['comm_id'] = \
                    self._analysises.get_free_proc_comm_id()

            proc = gui_subprocesses.Analysis(self._tc, **new_analsysis)
            self._analysises.push(proc)

            #Update live project status
            proc.set_callback_parameters(self._set_analysis_started)

        #
        #   3. CHECKING IF ANY ANALYSIS IS DONE
        #

        for analysis in self._analysises:

            analysis.set_callback_is_alive(
                self._analysis_is_alive)

            analysis.update()

        #
        #   4. TOGGLE SAVE STATUS OF SUBPROCESSES (if any is running)
        #

        if (self._analysises.count() == 0 and
                self._experiments.count() == 0 and
                self._queue.count() == 0):

            self.set_saved()

        else:

            self.set_unsaved()

        #UPDATE FREE SCANNERS
        sm['free-scanners'] = tc.scanners.count()

        #UPDATE LIVE PROJECTS
        sm['live-projects'] = self._project_progress.get_project_count()

        #UPDATE SUMMARY TABLE
        self._view.update()

        return True

    def _set_stopped_experiment(self, prefix):

        self.set_project_progress(
            prefix, 'EXPERIMENT', 'TERMINATED')

    def _set_stopped_analysis(self, prefix):

        self.set_project_progress(
            prefix, 'ANALYSIS', 'TERMINATED')

        self.set_project_progress(
            prefix, 'INSPECT', 'LAUNCH')

    def stop_process(self, proc):
        """Stops a process"""

        if (proc.get_type() == subproc_interface.EXPERIMENT_SCANNING or
                proc.get_type() == subproc_interface.EXPERIMENT_REBUILD):

            handler = self._experiments
            proc.set_callback_prefix(self._set_stopped_experiment)

        elif (proc.get_type() == subproc_interface.ANALYSIS):

            handler = self._analysises
            proc.set_callback_prefix(self._set_stopped_analysis)

        handler.remove(proc)
        proc.terminate()
        proc.close_communications()
