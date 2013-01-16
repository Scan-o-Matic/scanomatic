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

    def __init__(self, stdin, stdout):

        self.stdin = stdin
        self.stdout = stdout

    def poll(self):

        retval = 0
        t_string = ''
        try:
            fh = open(self.stdin, 'a')
            t_string = "__ECHO__ {0}\n".format(time.time())
            fh.write(t_string)
            fh.close()
        except:
            pass

        time.sleep(0.55)

        try:
            fh = open(self.stdout, 'r')
            lines = fh.read()
            if t_string in lines:
                retval = None
            fh.close()
        except:
            pass

        return retval


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
                stdin = paths.experiment_stdin.format(scanner)
                alive = True
                #try:
                fh = open(stdin, 'a')
                test_str = "__ECHO__ {0}\n".format(random.random())
                fh.write(test_str)
                fh.close()
                #except:
                #    alive = False

                if alive:

                    logger.info("Scanner {0} is alive".format(scanner_i))
                    self._check_scanner(
                        paths.log_scanner_out.format(scanner_i),
                        scanner_i, scanner, lines, test_str, 0)

                if not alive:
                    self._clean_after(scanner_i, scanner, lines)

        #CLEAING OUT PAD UUIDS NOT IN USE ACCORDING TO LOCKFILES
        try:
            fh = open(paths.lock_power_up_new_scanner, 'r')
            lines = fh.readlines()
            fh.close()
        except:
            lines = []

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() in ids:
                del lines[i]

        try:
            fh = open(paths.lock_power_up_new_scanner, 'r')
            fh.writelines(lines)
            fh.close()
        except:
            pass

    def _check_scanner(self, stdout_path, scanner_i, scanner, scanner_id, 
        test_str, try_x):

        tc = self.get_top_controller()
        paths = tc.paths
        alive = True
        try:
            fh = open(stdout_path, 'r')
            lines = fh.read()
            fh.close()
        except:
            alive = False

        if alive and test_str in lines:
            #DO STUFF TO REVIVE - NOT DONE
            stdin_path = paths.experiment_stdin.format(scanner)
            stderr_path = paths.log_scanner_err.format(scanner_i)
            proc = Fake_Proc(stdin_path, stdout_path)
            psm = model_experiment.copy_model(
                model_experiment.specific_project_model)

            self.add_subprocess(proc, 'scanner', stdin=stdin_path,
                stdout=stdout_path, stderr=stderr_path,
                pid=None, psm=psm,
                proc_name="Scanner {0}".format(scanner_i))

        else:

            if try_x < 10:
                try_x += 1
                gobject.timeout_add(100, self._check_scanner, stdout_path, scanner_i,
                    scanner, scanner_id, test_str, try_x)
            else:
                self._clean_after(scanner_i, scanner, scanner_id)

        return False

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

    def add_subprocess(self, proc, proc_type, stdin=None, stdout=None, stderr=None,
                        pid=None, psm=None, proc_name=None):

        sm = self._specific_model
        if proc_type == 'scanner':

            plist = sm['scanner-procs']

        elif proc_type == 'analysis':

            plist = sm['analysis-procs']

        else:

            raise Unknown_Subprocess_Type(proc_type)

        plist.append({'proc': proc, 'type': proc_type, 'pid': pid, 'stdin': stdin,
            'stdout': stdout, 'stderr': stderr, 'sm': psm, 'name': proc_name,
            'progress': 0, 'start-time': time.time()})

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
                        tc.paths.experiment_first_pass_analysis_relative))

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
                        stdout=stdout_path, stderr=stderr_path, sm=psm,
                        proc_name=proc_name)

                else:  # Report problem to user!
                    pass

                self._drop_process(p)
                scanner = p['sm']['scanner']
                #tc.scanners.free(scanner)

            else:

                lines = self._get_stdout_since_last_time(p)
                i_started = re.findall(r'__Is__ (.*)$', lines) 
                i_done = re.findall(r'__Id__ (.*)$', lines) 

                if len(i_done) > 0:

                    p['progress'] = int(i_done[-1])

                if len(i_started) > 0 and \
                    int(p['progress']) < int(i_started[-1]):

                    p['progress'] += 0.5


        #CHECK FOR TERMINATED ANALYSIS
        for p in self.get_subprocesses(by_type='analysis'):
            p_exit = p['proc'].poll()
            if p_exit is not None:

                self._drop_process(p)

                #DO A WARNING HERE SINCE NOT NICE QUIT!
                if p_exit != 0:
                    pass

            else:

                lines = self._get_stdout_since_last_time(p)
                progress = re.findall(r'INFO: __Is__ ([0-9]*)', lines)
                total = re.findall(r'INFO: ANALYSIS: A total of ([0-9]*)', lines)

                if len(total) > 0:
                    p['progress-total-number'] = int(total[0])

                if len(progress) > 0 and 'progress-total-number' in p:
                    p['progress'] = (float(progress[-1]) - 1) / \
                        p['progress-total-number']
                    if '1' in progress:
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

    def _get_stdout_since_last_time(self, p):

        lines = ""
        try:
            fh = open(p['stdout'], 'r')
            if 'stdout-pos' in p:
                fh.seek(p['stdout-pos'])
            lines = fh.read()
            p['stdout-pos'] = fh.tell()
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

