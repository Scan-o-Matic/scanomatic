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
import re
from subprocess import Popen

#
# INTERNAL DEPENDENCIES
#

import src.controller_generic as controller_generic
import src.view_subprocs as view_subprocs
import src.model_subprocs as model_subprocs

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

class Subprocs_Controller(controller_generic.Controller):

    def __init__(self, main_controller, logger=None):

        super(Subprocs_Controller, self).__init__(main_controller,
            specific_model=model_subprocs.get_composite_specific_model(),
            logger=logger)

        gobject.timeout_add(1000, self._subprocess_callback)

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
            'progress': 0})

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

                    proc_name = os.sep(psm['experiments-root'], 
                        psm['experiment-prefix'],
                        tc.paths.experiment_first_pass_analysis_relative)

                    a_dict['-i'] = proc_name

                    a_list = list()
                    for aflag, aval in a_dict.items():
                        a_list += [aflag, aval]

                    #START NEW PROC
                    proc = Popen(map(str, analysis_query), 
                        stdout=analysis_log, shell=False)

                    pid = proc.pid

                    analysis_log = open(os.sep.join(psm['experiments-root'],
                        psm['experiment-prefix'],
                        tc.paths.experiment_analysis_file_name) , 'w')

                    self.add_subprocess(proc, 'analysis', pid=pid,
                        stdout=analysis_log, sm=psm,
                        proc_name=proc_name)

                else:  # Report problem to user!
                    pass

                self._drop_process(p)
                scanner = p['sm']['scanner']
                tc.scanners.free(scanner)

            else:

                try:
                    fh = open(p['stdout'])
                    if 'stdout-pos' in p:
                        fh.seek(p['stdout-pos'])
                    lines = fh.read()
                    p['stdout-pos'] = fh.tell()
                    i_started = re.findall(r'__Is__ (.*)$', lines) 
                    i_done = re.findall(r'__Id__ (.*)$', lines) 
                    p['progress'] += len(i_done)
                    p['progress'] += 0.5 * (len(i_started) != len(i_done))
                    fh.close()
                except:
                    pass


        #CHECK FOR TERMINATED ANALYSIS
        for p in self.get_subprocesses(by_type='analysis'):
            p_exit = p['proc'].poll()
            if p_exit is not None:

                self._drop_process(p)

                #DO A WARNING HERE SINCE NOT NICE QUIT!
                if p_exit != 0:
                    pass

        #UPDATE FREE SCANNERS
        sm['free-scanners'] = tc.scanners.count()

        #UPDATE SUMMARY TABLE
        self._view.update()

        return True

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
        elif p['type'] == 'analysis':
            plist = sm['analysis-procs']    

        for i, proc in enumerate(plist):

            if id(p) == id(proc):

                try:
                    fs = open(p['stdin'], 'a')
                    fs.write("__QUIT__\n")
                    fs.close()
                except:
                    self._logger.error("Scanner won't be freed!")
                del plist[i]
                return True

        raise Unknown_Subprocess("{0}".format(p))

    def produce_running_experiments(self, widget):

        self.get_top_controller().add_contents_from_controller(
            view_subprocs.Running_Experiments(self, self._model,
            self._specific_model), 
            self._model['running-experiments'], 
            self)

    def produce_running_analysis(self, widget):

        pass
