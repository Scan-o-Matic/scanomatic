#!/usr/bin/env python
"""This module contains a class for obtaining images using SANE (Linux)."""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.993"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

from PIL import Image, ImageWin

import os, os.path, sys
from subprocess import call, Popen
import time
import types

#
# CLASSES
#

class Sane_Base():

    def __init__(self, owner=None, model=None, scan_mode=None):
        self.owner = owner
        self.next_file_name = None
        self._scanner_name = None
        self._program_name = "scanimage"
        self._scan_settings = None

        self._scan_settings_repo = \
            {"EPSON V700" : \
                {'TPU': \
                    ["--source", "Transparency" ,"--format", "tiff", 
                    "--resolution", "600", "--mode", "Gray", "-l", "0",  
                    "-t", "0", "-x", "203.2", "-y", "254", "--depth", "8"],
                'COLOR': \
                    ["--source", "Flatbed", "--format", "tiff",
                    "--resolution", "300", "--mode", "Color", "-l", "0",
                    "-t", "0", "-x", "215.9", "-y", "297.18", "--depth", "8"]} }
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

    def AcquireNatively(self, scanner=None, handle=None):
        return self.AcquireByFile(scanner=None, handle=handle)

    def AcquireByFile(self, scanner=None, handle=None):
        if self.next_file_name:
            self.owner.owner.DMS("Scanning", str(self.next_file_name), level=1)
            #os.system(self._scan_settings + self.next_file_name) 
            
            try:
                im = open(self.next_file_name,'w')
            except:
                self.owner.owner.DMS("ERROR", "Could not write to file: " + str(self.next_file_name),
                    level=1110)
                return False

            scan_query = list(self._scan_settings)
            if scanner != None:
                scan_query = ['-d', scanner] + scan_query
  
            scan_query.insert(0, self._program_name)  

            if self.owner and self.owner.USE_CALLBACK:
                return ("SANE-CALLBACK", im, Popen(scan_query, stdout=im, shell=False),self.next_file_name)
            else:
                call(scan_query,  stdout=im, shell=False) 

                im.close()

                return True

        else:
            return False
