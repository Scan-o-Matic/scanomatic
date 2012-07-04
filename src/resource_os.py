#!/usr/bin/env python
"""Resource module for inter-platform compatibility"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.994"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os, os.path, sys

#
# CLASSES
#

class OS_Settings():
    def __init__(self):
        name = None
        sort = -1
        attributeslist_length = 3
        attributes = {}

    def __getitem__(self, key):
        try:
            return self.attributes[key][self.sort]
        except:
            if key == "name":
                return self.name
            else:
                return None
 
    def __setitem__(self, key, value):
        try:
            self.attributes[key][self.sort] = value
            return True
        except:
            return False

    def set_attribute(self, key, attributeslist): 
        if type(attributeslist) == types.ListType and\
                len(attributeslist) == self.attributeslist_length:

            self.attributes[key] = []
            for value in attributeslist:
                self.attributes[key].append(value)
            return True

        else:
            return False 

class OS(OS_Settings):
    def __init__(self):
        OS_Settings.__init__(self)

        if os.name == "posix":
            self.name = 'linux'
            self.sort = 0
        elif os.name == "mac":
            self.name = "mac"
            self.sort = 1
        elif os.name == "nt":
            self.name = 'windows'
            self.sort = 2
        else:
            self.sort = -2

    def get_Free_Space(self, path=None):
        if not path:
            path = os.getcwd()

        try:
            d = os.statvfs(path)
            return d.f_bsize * d.f_bavail
        except:
           return -2

    def get_Is_Sufficient_Free_Space(self, path, copies=1, safty_factor=1.1):
        try:
            filesize = os.path.getsize(path)
        except:
            filesize = -1

        if filesize > 0:
            space_needed = filesize * copies


            space_on_drive = get_Free_Space(path)
            print filesize, copies, space_needed, space_on_drive
            return space_needed * safty_factor < space_on_drive
        else:
            return filesize

