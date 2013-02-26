#!/usr/bin/env python
"""Relaunch script for non-terminated experiments and analysis"""
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

import os
import time
from subprocess import Popen, PIPE

#
# INTERNAL DEPENDENCIES
#

import src.resource_path as resource_path
import src.resource_app_config as resource_app_config
import src.resource_logger as resource_logger
import src.resource_scanner as resource_scanner

class Locator(object):

    def __init__(self, log_path=None):

        self._paths = resource_path.Paths()
        self._logger = resource_logger.File_Logger(
            path=self._paths.log_relaunch)
        self._app_config = resource_app_config.Config(paths=self._paths)

    def run(self):

        experiment_files = []
        analysis_files = []

        curdir = self._paths.experiment_root
        e_pattern = self._paths.experiment_first_pass_analysis_relative[3:]
        a_pattern = self._paths.analysis_run_log

        def _search(curdir):

            files = os.listdir(curdir)
            files = [os.path.join(curdir, f) for f in files]
            for f in files:
                if f.endswith(e_pattern) and os.path.isfile(f):
                    experiment_files.append(f)
                elif f.endswith(e_pattern) and os.path.isfile(f):
                    analysis_files.append(f)
                elif os.path.isdir(f):
                        _search(f)

        _search(os.path.abspath(curdir))

        self._kill_orphan_scanners()
        self._revive_experiments(experiment_files)
        self._revive_analysis(analysis_files)

    def _kill_orphan_scanners(self):

        scanners = resource_scanner.Scanners(self._paths, self._app_config, logger=self._logger)
        names = scanners.get_names(available=False)
        statuses = []

        for name in names:
            scanner = scanners[name]
            if scanner.get_claimed() == False:
                scanner.off()
            statuses.append(scanner.get_power_status() in (None, True))

        scanners.update()

        #IF THERE ARE ANY POTENTIALLY ON PM-SOCKETS
        if True in statuses:

            if scanners.get_names() == scanners.get_names(available=False):

                p = Popen('ps -A | grep python -c', stdout=PIPE, stderr=PIPE,
                    shell=True)

                python_procs, stderr = p.communicate()

                if python_procs.strip() != '1':

                    self._logger.warning('Scanners are on but should not be and could not be turned off safly')

                else:

                    for name in names:
                        scanner = scanners[name]
                        scanner.off(byforce=True)


    def _get_scanner_locks(self):
        """Returns a list of all UUIDs currently locking scanners"""
        lock_files = [self._paths.lock_scanner_pattern.format(s) 
            for s in range(self._app_config.number_of_scanners)]

        scanner_locks = {}
        for i, lf in enumerate(lock_files):
            try:
                fh = open(lf, 'r')
                sl = fh.read().strip()
                if len(sl) > 0:
                    scanner_locks[sl] = i
                fh.close()
            except:
                pass

        self._logger.info("Current scanner locks are: {0}".format(scanner_locks))
        return scanner_locks

    def _match_uuids(self, scanner_locks, experiment_files):
        
        revive_experiments = []

        if len(scanner_locks) > 0:
            for e in experiment_files:
                try:
                    fh = open(e,'r')
                    e_settings = eval(fh.readline().strip())
                    images = 0
                    for line in fh:
                        images += 1

                    if (e_settings is not None and 'UUID' in e_settings and
                            e_settings['UUID'] in scanner_locks):

                        revive_experiments.append([e_settings,
                                scanner_locks[e_settings['UUID']], images, e])

                    fh.close()
                except:
                    pass

        self._logger.info("The following experiments were found matching locks: {0}".format(revive_experiments))
        return revive_experiments


    def _select_dupes(self, dupelist):

        candidates = []
        for i, dupe in enumerate(dupelist):
            #Check if all images have been taken as requested or if
            #Time passed since start is less than total time
            if not(dupe[1][0]['Measures'] <= dupe[1][2] or
                    dupe[1][0]['Start Time'] != 0 and 
                    dupe[1][0]['Start Time'] + dupe[1][0]['Measures'] * 
                    dume[1][0]['Interval']*60 < time.time()):

                dupelist[i][1] = True  # is a duplicate

            else:

                candidates.append(dupe)
                dupelist[i][1] = False  # is not a duplicate

        #If we have more than one candidate, lets forget it all.
        if len(candidates) > 1:
            for i in range(len(dupelist)):
                dupelist[i][1] = True

        return dupelist

    def _filter_experiment_dupes(self, revive_experiments):

        exp_per_scanner = [0] * self._app_config.number_of_scanners

        for revive_experiment in revive_experiments:
            exp_per_scanner[revive_experiment[1]] += 1

        for scanner, n_experiments in enumerate(exp_per_scanner):
            if n_experiments > 1:
                #We have duplicates, this removes them
                dupe_selection = self._select_dupe([(i, revive_experiment)
                    for i, revive_experiment in enumerate(revive_experiments)
                    if revive_experiment[1] == scanner])

                for dupe in dupeselection:
                    if dupe[1] == True:
                        del revive_experiments[dupe[0]]

        self._logger.info("After duplicate filter the following were still considered: {0}".format(revive_experiments))
        return revive_experiments

    def _revive_experiments(self, experiment_files):

        revive_experiments = []

        #Checking if uuid in scanner locks
        scanner_locks = self._get_scanner_locks()

        #Match UUIDs
        revive_experiments = self._match_uuids(scanner_locks, experiment_files)

        revive_experiments = self._filter_experiment_dupes(revive_experiments)

        if len(revive_experiments) < len(scanner_locks):

            self._logger.error("Not all experiments that might have been running could be revived")
            
        for e in revive_experiments:

            e_query_list = [self._paths.experiment, '-e', e[3]]

            stdin_path = self._paths.experiment_stdin.format(
                self._paths.get_scanner_path_name(e[0]['Scanner']))
            stdin = open(stdin_path, 'w')
            stdin.close()

            stdout_path = self._paths.log_scanner_out.format(e[1])
            stdout = open(stdout_path, 'w')
            stderr_path = self._paths.log_scanner_err.format(e[1])
            stderr = open(stderr_path, 'w')

            self._logger.info("Re-Launching experiment {0} ".format(e[3]))

            proc = Popen(e_query_list, stdout=stdout, stderr=stderr,
                shell=False)

    def _write_warning(self, msg):

        print msg

    def _revive_analysis(self, analysis_files):

        print "Reviving analysis not yet implemented so you should restart these:"
        print analysis_files

if __name__ == "__main__":

    l = Locator()
    l.run()
