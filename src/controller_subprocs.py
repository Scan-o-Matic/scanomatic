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
import types
import random
import re
import os
import time
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

class No_View_Loaded(Exception): pass
class Not_Yet_Implemented(Exception): pass
class Unknown_Subprocess_Type(Exception): pass
class Unknown_Subprocess(Exception): pass
class UnDocumented_Error(Exception): pass

#
# FUNCTIONS
#

#
# CLASSES
#

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

        super(Subprocs_Controller, self).__init__(main_controller,
            specific_model=model_subprocs.get_composite_specific_model(),
            logger=logger)

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

        return model_subprocs.get_gui_model()

    def ask_destroy(self):
        """This is to allow the fake destruction always"""
        return True

    def destroy(self):
        """Subproc is never destroyed, but its views always allow destruction"""
        pass

    def add_subprocess(self, proc, proc_type, stdin=None, stdout=None,
            stderr=None, pid=None, psm=None, proc_name=None,
            start_time=None, progress=None):

        sm = self._specific_model
        if proc_type == 'scanner':

            plist = sm['scanner-procs']

        elif proc_type == 'analysis':

            plist = sm['analysis-procs']

        else:

            raise Unknown_Subprocess_Type(proc_type)

        if start_time is None:
            start_time = time.time()

        plist.append({'proc': proc, 'type': proc_type, 'pid': pid, 'stdin': stdin,
            'stdout': stdout, 'stderr': stderr, 'sm': psm, 'name': proc_name,
            'progress': progress, 'start-time': start_time})

    def get_subprocesses(self, by_name=None, by_type=None):

        sm = self._specific_model

        if by_type is None:
            plist = sm['scanner-procs'] +  sm['analysis-procs']
        elif by_type in ['scanner', 'experiment']:
            plist = sm['scanner-procs']
        elif by_type == 'analysis':
            plist = sm['analysis-procs']
        else:
            raise Unknown_Subprocess_Type(proc_name)

        ret = [p for p in plist if (by_name is not None and p['name'] == by_name or True)]

        return ret

    def _subprocess_callback(self):

        sm = self._specific_model
        tc = self.get_top_controller()

        #CHECK FOR SCANNERS THAT ARE DONE
        for p in self.get_subprocesses(by_type='scanner'):
            p_exit = p['proc'].poll()
            if p_exit is not None:

                #PROCESS WAS TERMINATED
                if p_exit == 0:  # Nice quit implies analysis should be started

                    psm = p['sm']
                    a_dict = tc.config.get_default_analysis_query()

                    proc_name = os.sep.join((psm['experiments-root'], 
                        psm['experiment-prefix'],
                        tc.paths.experiment_first_pass_analysis_relative.format(
                        psm['experiment-prefix'])))

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

                    self.add_subprocess(proc, 'analysis', pid=pid,
                        stdout=stdout_path, stderr=stderr_path, psm=psm,
                        proc_name=proc_name)

                else:  # Report problem to user!
                    pass

                self._drop_process(p)
                scanner = p['sm']['scanner']
                #tc.scanners.free(scanner)

            else:

                lines = self._get_output_since_last_time(p, 'stdout')
                #i_started = re.findall(r'__Is__ (.*)$', lines) 
                i_done = re.findall(r'__Id__ (.*)', lines) 

                if len(i_done) > 0:

                    p['progress'] = int(i_done[-1]) 
            
                """
                if len(i_started) > 0 and \
                    int(p['progress']) < int(i_started[-1]):

                    p['progress'] += 0.5
                """

        #CHECK FOR TERMINATED ANALYSIS
        for p in self.get_subprocesses(by_type='analysis'):
            p_exit = p['proc'].poll()
            if p_exit is not None:

                self._drop_process(p)

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
                        p['progress-elapsed-time'] = (time.time() -
                            p['start-time']) - p['progress-init-time']
                    p['progress-current-image'] = int(progress[-1])

        #UPDATE FREE SCANNERS
        sm['free-scanners'] = tc.scanners.count()

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
            if type(f) == types.FileType:
                f.close()

    def stop_process(self, p):

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
            view_subprocs.Running_Experiments(self, self._model,
            self._specific_model), 
            self._model['running-experiments'], 
            self)

    def produce_free_scanners(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Free_Scanners(self, self._model,
            self._specific_model), self._model['free-scanners'],
            self)
    
    def produce_running_analysis(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Running_Analysis(self, self._model,
            self._specific_model), self._model['running-analysis'],
            self)

    def produce_errors_and_warnings(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Errors_And_Warnings(self, self._model,
            self._specific_model), self._model['collected-messages'],
            self)

    def produce_gridding_images(self, widget, p):

        proj_dir = p['sm']['analysis-project-log_file_dir']
        proj_prefix = proj_dir.split(os.sep)[-1]

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.View_Gridding_Images(self, self._model,
            p['sm']), proj_prefix, self)
