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
import re
import os
import time
import ConfigParser
from subprocess import Popen

#
# INTERNAL DEPENDENCIES
#

import src.controller_generic as controller_generic
import src.view_subprocs as view_subprocs
import src.model_subprocs as model_subprocs
import src.model_experiment as model_experiment

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


class Fake_Proc(object):

    def __init__(self, stdin, stdout, stderr, logger=None):

        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._logger = logger

    def poll(self):

        retval = 0

        t_string = "__ECHO__ {0}\n".format(time.time())
        lines = self._get_feedback(t_string)

        if t_string in lines:
            retval = None

        return retval

    def _get_feedback(self, c):

        try:
            fh = open(self.stdout, 'r')
            fh.read()
            fh_pos = fh.tell()
            fh.close()
        except:
            fh_pos = None

        try:
            fh = open(self.stdin, 'a')
            fh.write(c)
            fh.close()
        except:
            self._logger.error('Could not write to stdin')

        lines = ""
        i = 0

        #self._logger.info('stdout pos: {0}, sent to stdin: {1}'.format(
        #    fh_pos, c))

        while i < 10 and "__DONE__" not in lines:

            try:
                fh = open(self.stdout, 'r')
                if fh_pos is not None:
                    fh.seek(fh_pos)
                lines += fh.read()
                fh_pos = fh.tell()
                fh.close()
            except:
                self._logger.error('Could not read stdout')

            if "__DONE__" not in lines:
                time.sleep(0.1)
                i += 1

        #self._logger.info('stdout pos: {0}, got response: "{1}"'.format(
        #    fh_pos, lines))

        return lines

    def communicate(self, c):

        return self._get_feedback(c)


class Subprocs_Controller(controller_generic.Controller):

    def __init__(self, main_controller, logger=None):

        super(Subprocs_Controller, self).__init__(
            main_controller,
            specific_model=model_subprocs.get_composite_specific_model(),
            logger=logger)

        self._project_progress = Handle_Progress(main_controller.paths,
                                                 self._model)

        self._check_scanners()
        self._find_analysis_procs()

        gobject.timeout_add(1000, self._subprocess_callback)

    def _check_scanners(self):

        tc = self.get_top_controller()
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
                proc = Fake_Proc(stdin_path, stdout_path, stderr_path, logger=self._logger)

                if proc.poll() is None:

                    logger.info("Scanner {0} is alive".format(scanner_i))
                    self._revive_scanner(scanner, scanner_i, proc=proc)

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

    def _revive_scanner(self, scanner, scanner_i, proc=None):

        tc = self.get_top_controller()
        paths = tc.paths
        stdin_path = paths.experiment_stdin.format(scanner)
        stdout_path = paths.log_scanner_out.format(scanner_i)
        stderr_path = paths.log_scanner_err.format(scanner_i)

        if proc is None:
            proc = Fake_Proc(stdin_path, stdout_path, stderr_path, logger=self._logger)

        psm_in_text = proc.communicate("__INFO__")

        psm = model_experiment.copy_model(
            model_experiment.specific_project_model)

        psm_prefix = re.findall(r'__PREFIX__ (.*)', psm_in_text)
        if len(psm_prefix) > 0:
            psm['experiment-prefix'] = psm_prefix[0]

        psm_fixture = re.findall(r'__FIXTURE__ (.*)', psm_in_text)
        if len(psm_fixture) > 0:
            psm['fixture'] = psm_fixture[0]

        psm_scanner = re.findall(r'__SCANNER__ (.*)', psm_in_text)
        if len(psm_scanner) > 0:
            psm['scanner'] = psm_scanner[0]

        psm_root = re.findall(r'__ROOT__ (.*)', psm_in_text)
        if len(psm_root) > 0:
            psm['experiments-root'] = psm_root[0]

        psm_pinning = re.findall(r'__PINNING__ (.*)', psm_in_text)
        if len(psm_pinning) > 0:
            psm['pinnings-list'] = map(tuple, eval(psm_pinning[0]))

        psm_interval = re.findall(r'__INTERVAL__ (.*)', psm_in_text)
        if len(psm_interval) > 0:
            psm['interval'] = float(psm_interval[0])

        psm_scans = re.findall(r'__SCANS__ ([0-9]*)', psm_in_text)
        if len(psm_scans) > 0:
            psm['scans'] = int(psm_scans[0])

        psm_init_time = re.findall(r'__INIT-TIME__ (.*)', psm_in_text)
        if len(psm_init_time) > 0:
            start_time = float(psm_init_time[0])
        else:
            start_time = None

        psm_cur_image = re.findall(r'__CUR-IM__ ([0-9])', psm_in_text)
        if len(psm_cur_image) > 0:
            current_progress = int(psm_cur_image[0])
        else:
            current_progress = None

        if psm['interval'] is not None and psm['scans'] is not None:
            psm['duration'] = psm['interval'] * psm['scans'] / 60.0

        if 'experiment-prefix' in psm:
            self._logger.info('Revived experiment {0}'.format(
                psm['experiment-prefix']))

        self.add_subprocess(proc, 'scanner', stdin=stdin_path,
                            stdout=stdout_path, stderr=stderr_path,
                            pid=None, psm=psm,
                            proc_name="Scanner {0}".format(scanner_i),
                            start_time=start_time,
                            progress=current_progress)

    def _clean_after(self, scanner_i, scanner, scanner_id):

        tc = self.get_top_controller()
        scanner_id = scanner_id.strip()

        #FREE SCANNER
        scanner = tc.scanners["Scanner {0}".format(scanner_i)]
        scanner.set_uuid(scanner_id)
        scanner.free()

    def _find_analysis_procs(self):

        pass

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

        sm = self._specific_model
        return sm['images-in-queue']

    def add_subprocess(self, proc, proc_type, stdin=None, stdout=None,
                       stderr=None, pid=None, psm=None, proc_name=None,
                       start_time=None, progress=0):

        sm = self._specific_model

        _pp = self._project_progress
        paths = self.get_top_controller().paths
        analysis_path = os.path.join(
            paths.experiment_analysis_relative_path,
            paths.analysis_run_log)

        if proc_type == 'scanner':

            plist = sm['scanner-procs']

            _pp.add_project(
                psm['experiment-prefix'],
                os.path.join(
                    psm['experiments-root'],
                    psm['experiment-prefix']),
                analysis_path=analysis_path)

            _pp.set_status(psm['experiment-prefix'],
                           'EXPERIMENT', 'RUNNING')

        elif proc_type == 'analysis':

            plist = sm['analysis-procs']
            log_file_path = None
            if 'experiment-prefix' not in psm:

                proj_file = psm['analysis-project-log_file_dir']
                psm['experiments-root'] = os.path.abspath(
                    os.path.join(proj_file, os.pardir))

                psm['experiment-prefix'] = proj_file.split(os.path.sep)[-1]

                log_file_path = os.path.basename(psm['analysis-project-log_file'])

            if 'analysis-project-output-path' in psm:
                analysis_path = os.path.join(
                    psm['analysis-project-output-path'],
                    paths.analysis_run_log)

            _pp.add_project(
                psm['experiment-prefix'],
                os.path.join(
                    psm['experiments-root'],
                    psm['experiment-prefix']),
                first_pass_file=log_file_path,
                analysis_path=analysis_path)

            if _pp.get_status(psm['experiment-prefix'],
                              'EXPERIMENT', as_text=False) == 0:

                _pp.set_status(psm['experiment-prefix'],
                               'EXPERIMENT', 'COMPLETED')

            _pp.set_status(psm['experiment-prefix'],
                           'ANALYSIS', 'RUNNING')

        else:

            raise Unknown_Subprocess_Type(proc_type)

        if start_time is None:
            start_time = time.time()

        plist.append({'proc': proc, 'type': proc_type, 'pid': pid,
                      'stdin': stdin, 'stdout': stdout, 'stderr': stderr,
                      'sm': psm, 'name': proc_name, 'progress': progress,
                      'start-time': start_time})

        self.set_unsaved()

    def get_subprocesses(self, by_name=None, by_type=None):

        sm = self._specific_model

        if by_type is None:
            plist = sm['scanner-procs'] + sm['analysis-procs']
        elif by_type in ['scanner', 'experiment']:
            plist = sm['scanner-procs']
        elif by_type == 'analysis':
            plist = sm['analysis-procs']
        else:
            raise Unknown_Subprocess_Type("")

        ret = [p for p in plist if (by_name is not None and p['name'] == by_name or True)]

        return ret

    def _subprocess_callback(self):

        sm = self._specific_model
        tc = self.get_top_controller()
        sm['images-in-queue'] = 0

        _pp = self._project_progress

        #CHECK FOR SCANNERS THAT ARE DONE
        for p in self.get_subprocesses(by_type='scanner'):
            p_exit = p['proc'].poll()
            if p_exit is not None:

                #PROCESS WAS TERMINATED
                if p_exit == 0:  # Nice quit implies analysis should be started

                    psm = p['sm']

                    _pp.set_status(psm['experiment-prefix'],
                                   'EXPERIMENT', 'COMPLETED')

                    _pp.set_status(psm['experiment-prefix'],
                                   'ANALYSIS', 'AUTOMATIC')

                    a_dict = tc.config.get_default_analysis_query()

                    proc_name = os.path.join(
                        psm['experiments-root'],
                        psm['experiment-prefix'],
                        tc.paths.experiment_first_pass_analysis_relative.format(
                            psm['experiment-prefix']))

                    a_dict['-i'] = proc_name

                    a_list = list()
                    a_list.append(tc.paths.analysis)

                    for aflag, aval in a_dict.items():
                        a_list += [aflag, aval]

                    #START NEW PROC
                    stdout_path, stderr_path = tc.paths.get_new_log_analysis()
                    stdout = open(stdout_path, 'w')
                    stderr = open(stderr_path, 'w')

                    proc = Popen(map(str, a_list), stdout=stdout, stderr=stderr, shell=False)
                    print "Starting Analysis {0}".format(a_list)

                    pid = proc.pid

                    self.add_subprocess(
                        proc, 'analysis', pid=pid, stdout=stdout_path,
                        stderr=stderr_path, psm=psm, proc_name=proc_name)

                else:  # Report problem to user!
                    pass

                self._drop_process(p)
                #tc.scanners.free(scanner)

            else:

                lines = self._get_output_since_last_time(p, 'stdout')
                #i_started = re.findall(r'__Is__ (.*)$', lines)
                i_done = re.findall(r'__Id__ (.*)', lines)

                if len(i_done) > 0:

                    p['progress'] = int(i_done[-1])

                #COUNTING TOTAL SUM OF IMAGES TO TAKE
                sm['images-in-queue'] += psm['scans'] - p['progress']

                """
                if len(i_started) > 0 and \
                    int(p['progress']) < int(i_started[-1]):

                    p['progress'] += 0.5
                """

        #CHECK FOR TERMINATED ANALYSIS
        for p in self.get_subprocesses(by_type='analysis'):
            p_exit = p['proc'].poll()
            psm = p['sm']
            if p_exit is not None:

                self._drop_process(p)

                if p_exit == 0:
                    _pp.set_status(psm['experiment-prefix'],
                                   'ANALYSIS', 'COMPLETED')
                    _pp.set_status(psm['experiment-prefix'],
                                   'UPLOAD', 'LAUNCH')
                else:
                    _pp.set_status(psm['experiment-prefix'],
                                   'ANALYSIS', 'FAILED')

                #DO A WARNING HERE SINCE NOT NICE QUIT!
                if p_exit != 0:
                    pass

            else:

                lines = self._get_output_since_last_time(p, 'stderr')
                progress = re.findall(r'__Is__ ([0-9]*)', lines)
                total = re.findall(r'A total of ([0-9]*)', lines)

                if len(total) > 0:

                    p['progress-total-number'] = int(total[0])

                if len(progress) > 0 and 'progress-total-number' in p:
                    p['progress'] = (float(progress[-1]) - 1) / \
                        p['progress-total-number']
                    if 'progress-init-time' not in p:
                        p['progress-init-time'] = time.time() - p['start-time']
                    if progress != ['1']:
                        p['progress-elapsed-time'] = \
                            (time.time() - p['start-time']) - \
                            p['progress-init-time']

                    p['progress-current-image'] = int(progress[-1])

                if 'progress-current-image' in p:

                    _pp.set_status(psm['experiment-prefix'],
                                   'INSPECT', 'LAUNCH')

        #SET SAVED IF NO PROCS RUNNING
        if len(self.get_subprocesses()) == 0:
            self.set_saved()

        #UPDATE FREE SCANNERS
        sm['free-scanners'] = tc.scanners.count()

        #UPDATE LIVE PROJECTS
        sm['live-projects'] = self._project_progress.get_project_count()

        #UPDATE SUMMARY TABLE
        self._view.update()

        return True

    def _get_output_since_last_time(self, p, feed):

        lines = ""
        try:
            fh = open(p[feed], 'r')
            if 'output-pos' in p:
                fh.seek(p['output-pos'])
            lines = fh.read()
            p['output-pos'] = fh.tell()
            fh.close()
        except:
            pass

        return lines

    def _close_proc_files(self, *args):

        for f in args:
            if hasattr(f, 'close'):
                f.close()

    def stop_process(self, p):

        _pp = self._project_progress
        psm = p['sm']

        _pp.set_status(psm['experiment-prefix'],
                       'EXPERIMENT', 'TERMINATED')

        _pp.set_status(psm['experiment-prefix'],
                       'ANALYSIS', 'LAUNCH')

        self._drop_process(p)

    def _drop_process(self, p):

        sm = self._specific_model

        if p['type'] == 'scanner':
            plist = sm['scanner-procs']
            for i, proc in enumerate(plist):

                if id(p) == id(proc):

                    if p['proc'].poll() is None:
                        try:
                            fs = open(p['stdin'], 'a')
                            fs.write("__QUIT__\n")
                            fs.close()
                        except:
                            self._logger.error("Scanner won't be freed!")

                    del plist[i]
                    return True

        elif p['type'] == 'analysis':

            plist = sm['analysis-procs']

            if p['proc'].poll() is None:
                self._logger.info("Analysis will continue in the background...")
            else:
                self._logger.info("Analysis was complete")
                for i, proc in enumerate(plist):
                    if id(p) == id(proc):
                        del plist[i]
                        return True

            return True

        raise Unknown_Subprocess("{0}".format(p))

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
