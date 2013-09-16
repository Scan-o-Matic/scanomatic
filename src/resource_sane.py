#!/usr/bin/env python
"""This module contains a class for obtaining images using SANE (Linux)."""

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

from subprocess import call, Popen
import logging

#
# INTERNAL DEPENDENCIES
#


#
# CLASSES
#

class Sane_Base():

    def __init__(self, owner=None, model=None, scan_mode=None,
                 scan_settings=None):

        self.owner = owner

        self._logger = logging.getLogger("SANE")

        self.next_file_name = None
        self._scanner_name = None
        self._program_name = "scanimage"
        self._scan_settings = None

        if scan_settings is not None:
            self._scan_settings_repo = scan_settings
        else:
            self._scan_settings_repo = {
                "EPSON V700": {
                    'TPU': [
                        "--source", "Transparency", "--format", "tiff",
                        "--resolution", "600", "--mode", "Gray", "-l", "0",
                        "-t", "0", "-x", "203.2", "-y", "254", "--depth", "8"],
                    'COLOR': [
                        "--source", "Flatbed", "--format", "tiff",
                        "--resolution", "300", "--mode", "Color", "-l", "0",
                        "-t", "0", "-x", "215.9", "-y", "297.18",
                        "--depth", "8"]}}

        if model is not None:
            model = model.upper()

        if model not in self._scan_settings_repo.keys():
            self._model = self._scan_settings_repo.keys()[0]
        else:
            self._model = model
        if scan_mode is not None:
            scan_mode = scan_mode.upper()
            if scan_mode == "COLOUR":
                scan_mode = "COLOR"

        if scan_mode not in self._scan_settings_repo[self._model].keys():
            self._mode = self._scan_settings_repo[self._model].keys()[0]
        else:
            self._mode = scan_mode

        self._scan_settings = self._scan_settings_repo[self._model][self._mode]

    def OpenScanner(self, mainWindow=None, ProductName=None, UseCallback=False):
        pass

    def Terminate(self):
        pass

    def AcquireNatively(self, scanner=None, handle=None, filename=None):
        return self.AcquireByFile(scanner=None, handle=handle, filename=filename)

    def AcquireByFile(self, scanner=None, handle=None, filename=None):

        if filename is not None:
            self.next_file_name = filename

        if self.next_file_name:
            self._logger.info("Scanning {0}".format(self.next_file_name))
            #os.system(self._scan_settings + self.next_file_name)

            try:
                im = open(self.next_file_name, 'w')
            except:
                self._logger.error("Could not write to file: {0}".format(
                    self.next_file_name))
                return False

            scan_query = list(self._scan_settings)
            if scanner is not None:
                scan_query = ['-d', scanner] + scan_query

            scan_query.insert(0, self._program_name)
            self._logger.info("Scan-query is:\n{0}".format(" ".join(scan_query)))

            if self.owner is not None and self.owner.USE_CALLBACK:
                args = ("SANE-CALLBACK", im, Popen(scan_query, stdout=im, shell=False),
                        self.next_file_name)
                return args
            else:
                call(scan_query,  stdout=im, shell=False)

                im.close()

                return True

        else:
            return False
