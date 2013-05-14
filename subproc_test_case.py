#!/usr/bin/env python
"""Subprocess test-case is meant to be used to debug communications with subprocesses"""

import threading
import sys
from time import sleep

import src.subprocs.communicator as communicator
import src.resource_logger as resource_logger


class Test_Subprocess(object):

    def __init__(self, stdin, stdout, stderr):

        logger = resource_logger.Fallback_Logger()
        self._comm = communicator.Communicator(
            logger, self,  stdin, stdout, stderr)

        self._comm_thread = threading.Thread(target=self._comm.run)
        self._comm_thread.start()

        self._running = True
        self._paused = False
        self._current = 0
        self._total = 10000

    def run(self):

        while self._running:

            if self._current >= self._total:
                self._running = False

            if not self._paused:
                self._current += 1

            sleep(1)

    def get_current_step(self):

        return self._current

    def get_total_iterations(self):

        return self._total

    def get_progress(self):

        return float(self.get_current_step()) / self.get_total_iterations()

    def get_paused(self):

        return self._paused

    def set_terminate(self):

        self._running = False
        return True

    def set_pause(self):

        self._paused = True
        return True

    def set_unpause(self):

        self._paused = False
        return True

    def get_info(self):

        return ("__NAME__ Test Case",
                "__TOTAL__ {0}".format(self._total))

if __name__ == "__main__":

    if len(sys.argv) < 3:
        fpath = "_test_case.std"
    else:
        fpath = sys.argv[-1]

    main = Test_Subprocess(fpath + "in", fpath + "out", fpath + "err")
    main.run()
