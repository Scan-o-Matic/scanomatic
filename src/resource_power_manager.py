#!/usr/bin/env python
"""Resource module using a SIS-PM with the scanner to control its power."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.996"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import sys
import time
import urllib2
from urllib import urlencode

#
# CLASSES
#
class USB_PM(object):
    def __init__(self, path, on_string="", off_string=""):
        self.name ="USB connected PM"
        self._path = path
        self._on_string = on_string
        self._off_string = off_string

    def on(self):
        #print "*** Calling", self._path, self._on_string
        os.system(str(self._path)+' '+str(self._on_string))

    def off(self): 
        #print "*** Calling", self._path, self._off_string
        os.system(str(self._path)+' '+str(self._off_string))

class USB_PM_LINUX(USB_PM):

    def __init__(self, socket, path = "sispmctl"):

        self.name ="USB connected PM (Linux)"
        self._socket = socket
        self._path = path
        self._on_string = "-o %{0}".format(socket)
        self._off_string = "-f %{0}".format(socket)

class USB_PM_WIN(USB_PM):

    def __init__(self, socket, path = r"C:\Program Files\Gembird\Power Manager\pm.exe"):

        self.name ="USB connected PM (Windows)"
        self._socket = socket
        self._path = path
        self._on_string = "-on -PW1 -Scanner{0}".format(socket)
        self._off_string = "-off -PW1 -Scanner{0}".format(socket)

class LAN_PM(object):

    def __init__(self, host, socket, password):

        self.name ="LAN connected PM"
        self._host = host
        self._socket = socket
        self._password = password is not None and password or "1"

        self._pwd_params = urlencode((("pw", password),))
        self._on_params = urlencode((("cte{0}".format(socket),1),))
        self._on_params = urlencode((("cte{0}".format(socket),0),))

        self._login_out_url = "http://{0}/login.html".format(host)
        self._ctrl_panel_url = "http://{0}/".format(host)


    def _login(self):

        return urllib2.urlopen(self._login_out_url, self._pwd_params)

    def _logout(self):

        return urllib2.urlopen(self._login_out_url)

    def on(self):

        self._login()

        urllib2.urlopen(self._ctrl_panel_url, self._on_params)

        self._logout()

    def off(self):

        self._login()

        urllib2.urlopen(self._ctrl_panel_url, self._off_params)

        self._logout()

class Power_Manager():
    def __init__(self, pm=None, DMS=None):
        self._pm = pm 
        self._installed = pm is not None
        self._on = None
        if DMS is not None:
            self._DMS = DMS
        else:
            self._DMS = self.no_view

        if pm is not None:
            self._DMS("Power", "Hooked up {0}, Socket {1}".format(pm.name, 
                pm._socket))
        else:
            self._DMS("Power", "Power Manager has no device to talk to")

    def no_view(self, *args, **args2):
        pass

    def on(self):
        
        if self._installed and self._on != True:
            self._pm.on()
            self._on = True
            if self._DMS:
                self._DMS("Power","Switching on {0}, Socket {1}".format(
                    self._pm.name, self._pm._socket),
                    level="LA", debug_level='debug')     

    def off(self):
        if self._installed and self._on != False:
            self._pm.off()
            self._on = False
            if self._DMS:
                self._DMS("Power","Switching off {0}, Socket {1}".format(
                    self._pm.name, self._pm._socket),
                    level="LA", debug_level='debug')     

    def toggle(self):
        if self.on is None or self.on == False:
            self.on()
        else:
            self.off()

