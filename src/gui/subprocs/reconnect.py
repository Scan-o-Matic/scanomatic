#!/usr/bin/env python
"""The Subprocess-objects used by the mail program to communicate
with the true subprocesses"""
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


#
# INTERNAL DEPENDENCIES
#

import gui_subprocesses
import src.resource_logger as resource_logger

#
# CLASSES
#


class Reconnect_Subprocs(object):

    def __init__(self, controller, logger):

        self._controller = controller
        if logger is None:
            logger = resource_logger.Fallback_Logger()
        self._logger = logger
        self._tc = controller.get_top_controller()

    def run(self):

        self.check_scanners()
        self.check_analysises()

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

                proc = gui_subprocesses.Experiment_Scanning(
                    tc, **{
                        'stdin_path': stdin_path,
                        'stdout_path': stdout_path,
                        'stderr_path': stderr_path,
                        'is_running': True})

                if proc.is_done() is False:

                    logger.info("Scanner {0} is alive".format(scanner_i))

                    self._controller.add_subprocess_directly(
                        self._controller.EXPERIMENT_SCANNING, proc)

                else:

                    logger.info("Scanner {0} was dead".format(scanner_i))
                    self._release_scanner(scanner_i, scanner, lines)

        self._remove_uuids(ids)

    def check_analysises(self):

        pass

    def _remove_uuids(self, ids):

        paths = self._tc.paths
        logger = self._logger

        #CLEAING OUT BAD UUIDS NOT IN USE ACCORDING TO LOCKFILES
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

    def _release_scanner(self, scanner_i, scanner, scanner_id):

        scanner_id = scanner_id.strip()

        #FREE SCANNER
        scanner = self._tc.scanners["Scanner {0}".format(scanner_i)]
        scanner.set_uuid(scanner_id)
        scanner.free()
