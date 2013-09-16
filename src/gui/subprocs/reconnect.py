#!/usr/bin/env python
"""The Subprocess-objects used by the mail program to communicate
with the true subprocesses"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.999"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import inspect
import logging

#
# INTERNAL DEPENDENCIES
#

import src.gui.subprocs.communications.gui_communicator as gui_communicator
from src.gui.subprocs.event.event import Event

#
# METHODS
#


def whoCalled(fn):

    def wrapped(*args, **kwargs):
        frames = []
        frame = inspect.currentframe().f_back
        while frame.f_back:
            frames.append(inspect.getframeinfo(frame)[2])
            frame = frame.f_back
        frames.append(inspect.getframeinfo(frame)[2])

        print "===\n{0}\n{1}\n{2}\nCalled by {3}\n____".format(
            fn, args, kwargs, ">".join(frames[::-1]))

        fn(*args, **kwargs)

    return wrapped
#
# CLASSES
#


class Reconnect_Subprocs(object):

    def __init__(self, controller):

        self._paths = controller.get_top_controller().paths

        self._controller = controller
        self._logger = logging.getLogger("Reconnect Subprocs")
        self._tc = controller.get_top_controller()
        self._ids = {}

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
                    s_id = lines.split()[0].strip()
                else:
                    s_id = ""
                fh.close()
            except:
                locked = False

            logger.info("{0}: '{1}'".format(lock_path, lines))

            if locked:
                #TRY TALKING TO IT
                self._ids[scanner_i] = s_id
                logger.info("Scanner {0} is locked".format(scanner_i))

                stdin_path = paths.experiment_stdin.format(scanner)
                stdout_path = paths.log_scanner_out.format(scanner_i)
                stderr_path = paths.log_scanner_err.format(scanner_i)

                proc = gui_communicator.Experiment_Scanning(
                    tc, **{
                        'stdin_path': stdin_path,
                        'stdout_path': stdout_path,
                        'stderr_path': stderr_path,
                        'is_running': True})

                self._controller.add_event(Event(
                    proc.set_callback_is_alive,
                    self._is_alive, False,
                    responseTimeOut=10))
            else:
                ids.append(ids)

        self._remove_uuids(ids)

    def _is_alive(self, proc, is_alive):

        logger = self._logger
        if is_alive:

            logger.info("Proc {0} {1} is alive".format(
                proc.get_type(), proc.get_sending_path()))

            self._controller.add_subprocess_directly(proc.get_type(), proc)

        else:

            logger.info("Proc {0} {1} was dead".format(
                proc.get_type(), proc.get_sending_path()))

            if proc.get_type() == gui_communicator.EXPERIMENT_SCANNING:

                self._release_scanner(proc)

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

    def _release_scanner(self, proc):

        scanner_i = self._paths.get_scanner_index(proc.get_sending_path())
        scanner_id = self._ids[scanner_i].strip()

        #FREE SCANNER
        scanner = self._tc.scanners["Scanner {0}".format(scanner_i)]
        scanner.set_uuid(scanner_id)
        scanner.free()
