#!/usr/bin/env python
"""Resource module using a SIS-PM with the scanner to control its power."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.995"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os, os.path, sys
import time

#
# CLASSES
#

class Power_Manager():
    def __init__(self, installed=False, path=None, on_string="", off_string="", DMS = None):
        self._installed = installed
        self._path = path
        self._on_string = on_string
        self._off_string = off_string
        self._on = None
        if DMS is not None:
            self._DMS = DMS
        else:
            self._DMS = self.no_view

    def no_view(self, *args, **args2):
        pass

    def on(self):
        
        if self._installed and self._on != True:
            #print "*** Calling", self._path, self._on_string
            os.system(str(self._path)+' '+str(self._on_string))
            self._on = True
            if self._DMS:
                self._DMS("Power","Switching on",level="LA", debug_level='debug')     

    def off(self):
        if self._installed and self._on != False:
            #print "*** Calling", self._path, self._off_string
            os.system('"'+str(self._path)+'" '+str(self._off_string))
            self._on = False
            if self._DMS:
                self._DMS("Power","Switching off",level="LA", debug_level='debug')     

    def toggle(self):
        if self.on != None:
            if self.on == True:
                self.off()
            else:
                self.on()
        else:
            self.on()

