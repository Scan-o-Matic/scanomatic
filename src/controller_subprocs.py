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

    def __init__(self, window, main_controller):

        super(Subprocs_Controller, self).__init__(window, main_controller,
            specific_model=model_subprocs.get_composite_specific_model())


    def _get_default_view(self):

        return view_subprocs.Subprocs_View(self, self._model, self._specific_model)

    def _get_default_model(self):

        return model_subprocs.get_gui_model()

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
            'stdout': stdout, 'stderr': stderr, 'sm': psm, 'name': proc_name})

        #IF THIS IS THE ONLY PROC WE NEED TO START CALLBACKING
        if sm['running-scanners'] + sm['running-analysis'] == 1:

            gobject.timeout_add(1000, self._subprocess_callback)
        
    def get_subprocesses(self, by_name=None, by_type=None):

        sm = self._specific_model

        if by_type is None:
            plist = sm['scanner-procs'] +  sm['analysis-procs']
        elif by_type == 'scanner':
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

        #CHECK FOR TERMINATED ANALYSIS
        for p in self.get_subprocesses(by_type='analysis'):
            p_exit = p['proc'].poll()
            if p_exit is not None:

                self._drop_process(p)

                #DO A WARNING HERE SINCE NOT NICE QUIT!
                if p_exit != 0:
                    pass

        #UPDATE SUMMARY TABLE
        self._view.update()

        #IF ANY PROCS ARE ALIVE KEEP CHECKING
        if sm['running-scanners'] > 0 or sm['running-analysis'] > 0:
            self.set_unsaved()
            return True
        else:
            self.set_saved()
            return False

    def _close_proc_files(self, stdin, stdout, stderr):

        if type(stdin) == types.FileType:
            stdin.close()
        if type(stdout) == types.FileType:
            stdout.close()
        if type(stderr) == types.FileType:
            stderr.close()

    def _drop_process(self, p):

        sm = self._specific_model

        if p['type'] == 'scanner':
            plist = sm['scanner-procs']
        elif p['type'] == 'analysis':
            plist = sm['analysis-procs']    

        for i, proc in enumerate(plist):

            if id(p) == id(proc):

                self._close_proc_files(p['stdin'], p['stdout'], p['stderr'])
                del plist[i]
                return True

        raise Unknown_Subprocess("{0}".format(p))

    def produce_running_scanners(self, widget):

        pass

    def produce_running_analysis(self, widget):

        pass
