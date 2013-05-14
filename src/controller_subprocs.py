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
import os
import ConfigParser

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

#
# EXCEPTIONS
#


class No_View_Loaded(Exception):
    pass


class Not_Yet_Implemented(Exception):
    pass


class Unknown_Subprocess_Type(Exception):
    pass


class Unknown_Subprocess(Exception):
    pass


class UnDocumented_Error(Exception):
    pass


class InvalidStageOrStatus(Exception):
    pass


class InvalidProjectOrStage(Exception):
    pass

#
# FUNCTIONS
#

#
# CLASSES
#


class _Revive_Processes(object):

    def __init__(self, controller):

        self._controller = controller
        self._tc = controller.get_top_controller()

    def check_scanners(self):

        tc = self._tc
        paths = tc.paths
        config = tc.config
        ids = list()
        logger = self._logger

        for scanner_i in range(1, config.number_of_scanners + 1):

            logger.info("Checking scanner {0}".format(scanner_i))
            scanner = paths.get_scanner_path_name(
                config.scanner_name_pattern.format(scanner_i))

            lock_path = paths.lock_scanner_pattern.format(scanner_i)
            locked = False

            #CHECK LOCK-STATUS
            lines = ''
            try:
                fh = open(lock_path, 'r')
                lines = fh.read()
                if lines != '':
                    locked = True
                    ids.append(lines.split()[0].strip())
                fh.close()
            except:
                locked = False

            logger.info("{0}: {1}".format(lock_path, lines))

            if locked:
                #TRY TALKING TO IT
                logger.info("Scanner {0} is locked".format(scanner_i))
                stdin_path = paths.experiment_stdin.format(scanner)
                stdout_path = paths.log_scanner_out.format(scanner_i)
                stderr_path = paths.log_scanner_err.format(scanner_i)
                proc = gui_subprocesses.Experiment_Scanning(**{
                    'stdin_path': stdin_path,
                    'stdout_path': stdout_path,
                    'stderr_path': stderr_path,
                    'logger': logger})
                #proc = Fake_Proc(stdin_path, stdout_path, stderr_path, logger=self._logger)

                if proc.is_done() is None:

                    logger.info("Scanner {0} is alive".format(scanner_i))
                    self._revive_scanner(scanner, scanner_i, proc=proc)

                    self.add_subprocess(proc, 'scanner', stdin=stdin_path,
                                        stdout=stdout_path, stderr=stderr_path,
                                        pid=None, psm=psm,
                                        proc_name="Scanner {0}".format(scanner_i),
                                        start_time=start_time,
                                        progress=current_progress)

                else:

                    logger.info("Scanner {0} was dead".format(scanner_i))
                    self._clean_after(scanner_i, scanner, lines)

        #CLEAING OUT PAD UUIDS NOT IN USE ACCORDING TO LOCKFILES
        try:
            fh = open(paths.lock_power_up_new_scanner, 'r')
            lines = fh.readlines()
            fh.close()
        except:
            lines = []

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() not in ids:
                logger.info(
                    "Removing scanner uuid {0} from start-up queue".format(
                    lines[i].strip()))

                del lines[i]

        logger.info('Start-up queue is {0}'.format(lines))

        try:
            fh = open(paths.lock_power_up_new_scanner, 'w')
            fh.writelines(lines)
            fh.close()
        except:
            pass

    def find_analysis_procs():

        pass

    def _clean_after(self, scanner_i, scanner, scanner_id):

        tc = self.get_top_controller()
        scanner_id = scanner_id.strip()

        #FREE SCANNER
        scanner = tc.scanners["Scanner {0}".format(scanner_i)]
        scanner.set_uuid(scanner_id)
        scanner.free()


class Handle_Progress(object):

    #STAGES
    EXPERIMENT = 0
    ANALYSIS = 1
    INSPECT = 2
    UPLOAD = 3

    #STAGE STATUSES
    FAILED = -1
    NOT_YET = 0
    AUTOMATIC = 1
    LAUNCH = 2
    RUNNING = 3
    TERMINATED = 4
    COMPLETED = 5

    def __init__(self, paths, model):

        self._config = ConfigParser.ConfigParser(allow_no_value=True)
        self._paths = paths
        self._model = model

        self._count = 0
        self._load()

    def _load(self):

        try:
            self._config.read(self._paths.log_project_progress)
        except:
            pass

        self._count = len(self._config.sections())

    def _save(self):

        with open(self._paths.log_project_progress, 'wb') as configfile:
                self._config.write(configfile)

    def add_project(self, project_prefix, experiment_dir,
                    first_pass_file=None, analysis_path=None):

        if project_prefix not in self._config.sections():
            self._config.add_section(project_prefix)

        if first_pass_file is None:
            first_pass_file = \
                self._paths.experiment_first_pass_analysis_relative.format(
                    project_prefix)

        self._config.set(project_prefix, 'basedir', experiment_dir)
        self._config.set(project_prefix, '1st_pass_file', first_pass_file)
        self._config.set(project_prefix, 'analysis_path', analysis_path)
        self._config.set(project_prefix, str(self.EXPERIMENT), "0")
        self._config.set(project_prefix, str(self.ANALYSIS), "0")
        self._config.set(project_prefix, str(self.INSPECT), "0")
        self._config.set(project_prefix, str(self.UPLOAD), "0")
        self._save()

        self._count += 1

    def set_status(self, project_prefix, stage, status):

        if project_prefix not in self._config.sections():
            return False

        try:
            stage_num = eval("self." + stage)
            status_num = eval("self." + status)
        except:
            raise InvalidStageOrStatus("{0} {1}".format(stage, status))

        self._config.set(project_prefix, str(stage_num), str(status_num))
        self._save()

        if stage_num == self.UPLOAD and status_num == self.COMPLETED:
            self.clear_done_projects()

        return True

    def get_status(self, project_prefix, stage, as_text=True,
                   supress_load=False):

        if supress_load is False:
            self._load()

        try:
            if type(stage) == int:
                val = self._config.getint(project_prefix, str(stage))
            else:
                val = self._config.getint(
                    project_prefix,
                    str(eval("self." + stage)))

        except:
            raise InvalidProjectOrStage("{0} {1}".format(project_prefix, stage))

        if as_text:
            return self._model['project-progress-stage-status'][val]
        else:
            return val

    def get_all_status(self, project_prefix, as_text=True, supress_load=False):

        if supress_load is False:
            self._load()
        ret = []
        for stage in range(self.UPLOAD + 1):
            ret.append(self.get_status(project_prefix, stage,
                       as_text=as_text, supress_load=True))

        return ret

    def get_all_stages_status(self, as_text=True):

        self._load()
        ret = {}
        for project in self._config.sections():
            ret[project] = self.get_all_status(project, as_text=as_text,
                                               supress_load=True)

        return ret

    def remove_project(self, project_prefix, supress_save=False):

        self._config.remove_section(project_prefix)
        if supress_save is False:
            self._save()

    def clear_done_projects(self):

        projects = self.get_all_stages_status(as_text=False)

        for project, stages in projects.items():
            if not(False in [status > self.RUNNING for status in stages]):
                self.remove_project(project, supress_save=True)

        self._save()

    def get_project_count(self):

        return self._count

    def get_path(self, project, supress_load=False):

        if supress_load is False:
            self._load()
        return self._config.get(project, 'basedir')

    def get_analysis_path(self, project):

        self._load()
        analysis_path = os.path.join(
            self.get_path(project, supress_load=True),
            self._config.get(project, 'analysis_path'))

        return analysis_path

    def get_first_pass_file(self, project):

        self._load()
        first_pass_file = os.path.join(
            self.get_path(project, supress_load=True),
            self._config.get(project, '1st_pass_file'))

        return first_pass_file


class Subprocs_Controller(controller_generic.Controller):

    ANALYSIS = "A"
    EXPERIMENT_SCANNING = "ES"
    EXPERIMENT_REBUILD = "ER"

    def __init__(self, main_controller, logger=None):

        super(Subprocs_Controller, self).__init__(
            main_controller,
            specific_model=model_subprocs.get_composite_specific_model(),
            logger=logger)

        self._tc = self.get_top_controller()
        self._project_progress = Handle_Progress(main_controller.paths,
                                                 self._model)

        """
        revive_processes = _Revive_Processes(self)
        revive_processes.check_scanners()
        revive_processes.find_analysis_procs()
        """

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

        gobject.timeout_add(6421, self._subprocess_callback)

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

    def set_project_progress(self, prefix, stage, value):
        return self._project_progress.set_status(prefix, stage, value)

    def remove_live_project(self, prefix):

        self._project_progress.remove_project(prefix)

    def get_remaining_scans(self):

        img_tot = 0
        img_done = 0

        for proc in self._experiments:

            img_tot += proc.get_total()
            img_done += proc.get_current()

        return img_tot - img_done

    def add_subprocess(self, ptype, **params):
        """Adds a new subprocess.

        If it can be started, it will be immidiately, else
        if will be queued.

        **params should include sufficient information for the
        relevant gui_subprocess-class to launch
        """

        if ptype == self.EXPERIMENT_SCANNING:

            success = self._experiments.push(
                gui_subprocesses.Experiment_Scanning(self._tc, **params))

        elif ptype == self.ANALYSIS:

            success = self._queue.push(params)

        elif ptype == self.EXPERIMENT_REBUILD:

            success = self._experiments.push(
                gui_subprocesses.Experiment_Rebuild(self._tc, **params))

        self._logger.info("{0} {1} with parameters {2}".format(
            ["Failed to add", "Added"][success],
            ptype, params))

    def _subprocess_callback(self):
        """Callback that checks on finished stuff etc"""

        sm = self._specific_model
        tc = self._tc

        #
        #   1 CHECKING EXPERIMENTS IF ANY IS DONE
        #

        finished_experiment = self._experiments.pop()

        if finished_experiment is not None:
            #Place analysis in queue
            pass

            #Update live project status
            self.set_project_progress(
                finished_experiment.get_prefix(), 'EXPERIMENT', 'COMPLETED')

            self.set_project_progress(
                finished_experiment.get_prefix(), 'ANALYSIS', 'AUTOMATIC')

            finished_experiment.close_communications()

        #
        #   2. CHECKING IF ANY NEW ANALYSIS MAY BE STARTED
        #

        new_analsysis = self._queue.pop()

        if new_analsysis is not None:

            proc = gui_subprocesses.Analysis(self._tc, **new_analsysis)
            self._analysises.push(proc)
            #Update live project status
            self.set_project_progress(
                proc.get_prefix(), 'ANALYSIS', 'RUNNING')

        #
        #   3. CHECKING IF ANY ANALYSIS IS DONE
        #

        finished_analysis = self._analysises.pop()

        if finished_analysis is not None:

            if finished_analysis.get_exit_code() == 0:
                self.set_project_progress(
                    finished_analysis.get_prefix(),
                    'ANALYSIS', 'COMPLETED')
                self.set_project_progress(
                    finished_analysis.get_prefix(),
                    'UPLOAD', 'LAUNCH')
            else:
                self.set_project_progress(
                    finished_analysis.get_prefix(),
                    'ANALYSIS', 'FAILED')

            finished_analysis.close_communications()

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

    def stop_process(self, proc):
        """Stops a process"""

        if (proc.get_type() == subproc_interface.EXPERIMENT_SCANNING or
                proc.get_type() == subproc_interface.EXPERIMENT_REBUILD):

            handler = self._experiments
            cur_stage = 'EXPERIMENT'
            next_stage = 'ANALYSIS'

        elif (proc.get_type() == subproc_interface.ANALYSIS):

            handler = self._analysises
            cur_stage = 'ANALYSIS'
            next_stage = None

        handler.remove(proc)
        proc.terminate()

        self.set_project_progress(
            proc.get_prefix(), cur_stage, 'TERMINATED')

        if next_stage is not None:
            self.set_project_progress(
                proc.get_prefix(), next_stage, 'LAUNCH')

        proc.close_communications()

    def produce_running_experiments(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Running_Experiments(
                self, self._model,
                self._specific_model),
            self._model['running-experiments'],
            self)

    def produce_live_projects(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Live_Projects(self, self._model,
                                        self._specific_model),
            self._model['live-projects'],
            self)

    def produce_free_scanners(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Free_Scanners(self, self._model,
                                        self._specific_model),
            self._model['free-scanners'],
            self)

    def produce_running_analysis(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Running_Analysis(self, self._model,
                                           self._specific_model),
            self._model['running-analysis'],
            self)

    def produce_errors_and_warnings(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Errors_And_Warnings(self, self._model,
                                              self._specific_model),
            self._model['collected-messages'],
            self)

    def produce_inspect_gridding(self, widget, prefix, data={}):

        a_file = self._project_progress.get_analysis_path(prefix)

        data['stage'] = 'inspect'
        data['analysis-run-file'] = a_file
        data['project-name'] = prefix

        tc = self.get_top_controller()
        tc.add_contents(widget, 'analysis', **data)

    def produce_upload(self, widget, prefix):

        data = {'launch-filezilla': True}
        self.produce_inspect_gridding(widget, prefix, data=data)

    def produce_launch_analysis(self, widget, prefix):
        """produce_launch_analysis, short-cuts to displaying a
        view for analysing a specific project as defined in prefix
        """

        proj_dir = self._project_progress.get_path(prefix)
        data = {
            'stage': 'project',
            'analysis-project-log_file_dir': proj_dir,
            'analysis-project-log_file':
            self._project_progress.get_first_pass_file(prefix)}

        tc = self.get_top_controller()
        tc.add_contents(widget, 'analysis', **data)
