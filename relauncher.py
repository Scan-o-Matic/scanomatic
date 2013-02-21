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
                elif f.endswith(a.pattern) and os.path.isfile(f):
                    analysis:files.append(f)
                elif os.path.isdir(f):
                        _search(f)

        _search(os.path.abspath(dir))

        self._revive_experiments(experiment_files)
        self._revive_analysis(analysis_files)

    def _get_scanner_locks(self):
        """Returns a list of all UUIDs currently locking scanners"""
        lock_files = [self._paths.lock_scanner_pattern.format(s) 
            for s in range(self._app_config.scanners)]

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
        
        if len(scanner_locks) > 0:
            for e in experiment_files:
                try:
                    fh = open(e,'r')
                    e_settings = eval(fh.readline().strip())
                    if (e_settings is not None and 'UUID' in e_settings and
                            e_settings['UUID'] in scanner_locks):

                        revive_experiments.append([e_settings,
                                scanner_locks[e_settings['UUID']]])

                    fh.close()
                except:
                    pass

        return revive_experiments

    def _filter_experiment_dupes(self, revive_experiments):

        return revive_experiments

    def _revive_experiments(self, experiment_files):

        revive_experiments = []

        #Checking if uuid in scanner locks
        scanner_locks = self._get_scanner_locks()

        #Match UUIDs
        revive_experiments = self._match_uuids(scanner_locks, experiment_files)

        revive_experiments = self._filter_experiment_dupes(revive_experiments)

        if len(revive_experiments) < len(scanner_locks):

            pass
            
        for e in revive_experiments:

            pass

    def _revive_analysis(self, analysis_files):

        pass
