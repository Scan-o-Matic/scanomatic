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
import inspect

#
# INTERNAL DEPENDENCIES
#

import src.controller_generic as controller_generic
import view_subprocs
import model_subprocs

from src.gui.subprocs.handlers.analysis_queue import Analysis_Queue
import src.gui.subprocs.handlers.process_handler as process_handler

import src.gui.subprocs.communications.gui_communicator as gui_communicator
import src.gui.subprocs.progress_responses as progress_responses
import src.gui.subprocs.reconnect as reconnect
import src.gui.subprocs.live_projects as live_projects

import src.gui.subprocs.event.event_handler as event_handler
from src.gui.subprocs.event.event import Event

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
# METHODS
#


def whoCalled(fn):

    def wrapped(*args, **kwargs):
        frames = []
        frame = inspect.currentframe().f_back
        while frame.f_back:
            frames.append(inspect.getframeinfo(frame)[2])
            frame = frame.f_back
        frames.append(inspect.getframeinfo(frame)[2])

        print "===\n{0}\n{1}\n{2}\nCalled by {3}\n____".format(
            fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped

#
# CLASSES
#


class Subprocs_Controller(controller_generic.Controller,
                          progress_responses.Progress_Responses):

    def __init__(self, main_controller, logger=None):

        super(Subprocs_Controller, self).__init__(
            main_controller,
            specific_model=model_subprocs.get_composite_specific_model(),
            logger=logger)

        self._tc = self.get_top_controller()
        self._project_progress = live_projects.Live_Projects(
            main_controller.paths, self._model)

        #Initiate events handler and the timeout for its update
        self._subprocess_events = event_handler.EventHandler(logger)
        gobject.timeout_add(1307, self._subprocess_events.update)

        #Reconnect subprocesses from previous instances
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

        """ MULTI-EVENT
        img_tot = 0
        img_done = 0

        for proc in self._experiments:

            img_tot += proc.get_total()
            img_done += proc.get_current()

        return img_tot - img_done
        """
        pass

    def add_subprocess_directly(self, ptype, proc):
        """Adds a proc, should be used only by reconnecter"""

        if (ptype == gui_communicator.EXPERIMENT_SCANNING or
                ptype == gui_communicator.EXPERIMENT_REBUILD):

            success = self._experiments.push(proc)

            self.add_event(Event(
                proc.set_callback_parameters,
                self._set_experiment_started, None))

        elif ptype == gui_communicator.ANALYSIS:

            success == self._analysises.push(proc)

            self.add_event(Event(
                proc.set_callback_parameters,
                self._set_analysis_started, None))

        else:

            raise Exception("Proc {0} of type {1} is not known".format(
                proc, ptype))

            success = False

        return success

    def add_subprocess(self, ptype, **params):
        """Adds a new subprocess.

        If it can be started, it will be immidiately, else
        if will be queued.

        **params should include sufficient information for the
        relevant gui_subprocess-class to launch
        """

        if ptype == gui_communicator.EXPERIMENT_SCANNING:

            success = self.add_subprocess_directly(
                ptype,
                gui_communicator.Experiment_Scanning(self._tc, **params))

        elif ptype == gui_communicator.ANALYSIS:

            success = self._queue.push(params)

        elif ptype == gui_communicator.EXPERIMENT_REBUILD:

            if 'comm_id' not in params:
                params['comm_id'] = \
                    self._experiments.get_free_rebuild_comm_id()

            success = self.add_subprocess_directly(
                ptype,
                gui_communicator.Experiment_Rebuild(self._tc, **params))

        self._logger.info("{0} {1} with parameters {2}".format(
            ["Failed to add", "Added"][success],
            ptype, params))

    def add_event(self, event):
        """Adds a new event.

        :param event: The event
        """

        self._subprocess_events.addEvent(event)

    def stop_process(self, proc):
        """Stops a process

        :param proc: The process to be stopped
        """

        if (proc.get_type() == gui_communicator.EXPERIMENT_SCANNING or
                proc.get_type() == gui_communicator.EXPERIMENT_REBUILD):

            handler = self._experiments
            self.add_event(Event(
                proc.set_callback_prefix, self._set_stopped_experiment, None))

        elif (proc.get_type() == gui_communicator.ANALYSIS):

            handler = self._analysises
            self.add_event(Event(
                proc.set_callback_prefix, self._set_stopped_analysis, None))

        handler.remove(proc)
        proc.terminate()
        proc.close_communications()

    def _make_analysis_from_experiment(self, proc, param):
        """Queries and experiment process to run defualt analysis

        The analysis is placed in the queue and run with default
        parameters, only adjusting for the path to the specific
        project.

        :param proc: A finished experiment process.
        :return boolean: Success statement
        """

        if (param is not None and 'experiments-root' in param and
                'prefix' in param and param['prefix'] != '' and
                param['experiments-root'] != ''):

            self.add_subprocess(gui_communicator.ANALYSIS,
                                experiments_root=param['experiments-root'],
                                experiment_prefix=param['prefix'],
                                experiment_first_pass=param['1-pass file'])

            self._logger.info(
                ">>> Complete parameters converting experiment {0}".format(
                    param))

            return True

        else:

            self._logger.info(
                "<<< Incomplete parameters converting experiment {0}".format(
                    param))

        return False

    def _set_experiment_started(self, proc, param):

        if param is not None:
            self.set_project_progress(
                param['prefix'], 'EXPERIMENT', 'RUNNING',
                experiment_dir=param['experiments-root'],
                first_pass_file=param['1-pass file'])

    def _set_experiment_completed(self, proc, prefix):

        if prefix is not None:
            self.set_project_progress(prefix, 'EXPERIMENT', 'COMPLETED')
            self.set_project_progress(prefix, 'ANALYSIS', 'AUTOMATIC')
        else:
            self._logger.error("Failed to get the prefix for {0}".format(
                self._set_experiment_completed))

    def _set_analysis_started(self, proc, param):

            self.set_project_progress(
                param['prefix'], 'ANALYSIS', 'RUNNING',
                experiment_dir=param['experiments-root'],
                first_pass_file=param['1-pass file'],
                analysis_path=param['analysis-dir'])

    def _experiment_is_alive(self, experiment, is_alive):

        if is_alive is False and self._experiments.has(experiment):

            self._experiments.remove(experiment)

            #Place analysis in queue
            self.add_event(
                Event(experiment.set_callback_parameters,
                      self._make_analysis_from_experiment, None))

            #Update live project status
            self.add_event(
                Event(experiment.set_callback_prefix,
                      self._set_experiment_completed, None))

            experiment.close_communications()

    def _set_analysis_completed(self, proc, prefix):

        self.set_project_progress(
            prefix, 'ANALYSIS', 'COMPLETED')
        self.set_project_progress(
            prefix, 'UPLOAD', 'LAUNCH')

    def _set_analysis_failed(self, proc, prefix):

        self.set_project_progress(
            prefix, 'ANALYSIS', 'FAILED')

    def _analysis_is_alive(self, analysis, is_alive):

        if is_alive is False and self._analysises.has(analysis):

            self._analysises.remove(analysis)

            if analysis.get_exit_code() in (0, None):

                self.add_event(
                    Event(analysis.set_callback_prefix,
                          self._set_analysis_completed, None))
            else:

                self.add_event(
                    Event(analysis.set_callback_prefix,
                          self._set_analysis_failed, None))

            analysis.close_communications()

    def _subprocess_callback(self):
        """Callback that checks on finished stuff etc"""

        sm = self._specific_model
        tc = self._tc

        #
        #   1 CHECKING EXPERIMENTS IF ANY IS DONE
        #

        for experiment in self._experiments:

            self.add_event(Event(
                experiment.set_callback_is_alive,
                self._experiment_is_alive, False,
                responseTimeOut=10))

        #
        #   2. CHECKING IF ANY NEW ANALYSIS MAY BE STARTED
        #

        new_analsysis = self._queue.pop()

        if new_analsysis is not None:

            if 'comm_id' not in new_analsysis:
                new_analsysis['comm_id'] = \
                    self._analysises.get_free_proc_comm_id()

            proc = gui_communicator.Analysis(self._tc, **new_analsysis)
            self._analysises.push(proc)

            #Update live project status
            self.add_event(Event(
                proc.set_callback_parameters,
                self._set_analysis_started, None))

        #
        #   3. CHECKING IF ANY ANALYSIS IS DONE
        #

        for analysis in self._analysises:

            self.add_event(Event(
                analysis.set_callback_is_alive,
                self._analysis_is_alive, False,
                responseTimeOut=10))

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

    def _set_stopped_experiment(self, proc, prefix):

        self.set_project_progress(
            prefix, 'EXPERIMENT', 'TERMINATED')

    def _set_stopped_analysis(self, proc, prefix):

        self.set_project_progress(
            prefix, 'ANALYSIS', 'TERMINATED')

        self.set_project_progress(
            prefix, 'INSPECT', 'LAUNCH')
