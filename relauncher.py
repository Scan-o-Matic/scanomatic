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
from subprocess import Popen

#
# INTERNAL DEPENDENCIES
#

import src.resource_path as resource_path
import src.resource_app_config as resource_app_config

class Locator(object):

    def __init__(self):

        self._paths = resource_path.Paths()
        self._app_config = resource_app_config.Config(paths=self._paths)

    def run(self):

        experiment_files = []
        analysis_files = []

        dir = self._paths.experiment_root
        e_pattern = self._paths.experiment_first_pass_analysis_relative[3:]
        a_pattern = self._paths.analysis_run_log

        def _search(dir):

            files = os.listdir(dir)
            files = [os.path.join(dir, f) for f in files]
            for f in files:
                if f.endswith(e_pattern) and os.path.isfile(f):
                    experiment_files.append(f)
                elif f.endswith(e_pattern) and os.path.isfile(f):
                    analysis_files.append(f)
                elif os.path.isdir(f):
                        _search(f)

        _search(os.path.abspath(dir))

        self._revive_experiments(experiment_files)
        self._revive_analysis(analysis_files)

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

        return revive_experiments

    def _revive_experiments(self, experiment_files):

        revive_experiments = []

        #Checking if uuid in scanner locks
        scanner_locks = self._get_scanner_locks()

        #Match UUIDs
        revive_experiments = self._match_uuids(scanner_locks, experiment_files)

        revive_experiments = self._filter_experiment_dupes(revive_experiments)

        if len(revive_experiments) < len(scanner_locks):

            self._write_warnings("Not all experiments that were running could be revived")
            
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

            print "Launching experiment with", stdin, stdout, stderr

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
