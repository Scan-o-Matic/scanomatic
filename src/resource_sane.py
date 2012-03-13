#!/usr/bin/env python
"""This module contains a class for obtaining images using SANE (Linux)."""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.992"
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

    def __init__(self, owner=None):
        self.owner = owner
        self.next_file_name = None
        self._scanner_name = None
        self._program_name = "scanimage"
        self._scan_settings = ["--source", "Transparency" ,"--format", "tiff", "--resolution", "600", "--mode", "Gray", "-l", "0",  "-t", "0", "-x", "203.2", "-y", "254", "--depth", "8"]

    def OpenScanner(self, mainWindow=None, ProductName=None, UseCallback=False):
        pass

    def Terminate(self):
        pass

    def AcquireNatively(self, scanner=None, handle=None):
        return self.AcquireByFile()

    def AcquireByFile(self, scanner=None, handle=None):
        if self.next_file_name:
            self.owner.owner.DMS("Scanning", str(self.next_file_name), level=1)
            #os.system(self._scan_settings + self.next_file_name) 
            try:
                im = open(self.next_file_name,'w')
            except:
                self.owner.owner.DMS("ERROR", "Could not write to file: " + str(self.next_file_name),
                    level=1010)
                return False

            scan_query = list(self._scan_settings)
            scan_query.insert(0, self._program_name)
  
            if self.owner and self.owner.USE_CALLBACK:
                return ("SANE-CALLBACK", im, Popen(scan_query, stdout=im, shell=False),self.next_file_name)
            else:
                call(scan_query,  stdout=im, shell=False) 

                im.close()

                return True

        else:
            return False
