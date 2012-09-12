#!/usr/bin/env python
"""Resource module using a SIS-PM to control power to one socket at a time."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
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
import nmap
from subprocess import Popen, PIPE

#
# CLASSES
#

class USB_PM(object):
    """Base Class for USB-connected PM:s. Not intended to be used directly."""
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
    """Class for handling USB connected PM:s on linux."""
    def __init__(self, socket, path = "sispmctl"):

        self.name ="USB connected PM (Linux)"
        self._socket = socket
        self._path = path
        self._on_string = "-o {0}".format(socket)
        self._off_string = "-f {0}".format(socket)

class USB_PM_WIN(USB_PM):
    """Class for handling USB connected PM:s on windows."""
    def __init__(self, socket, path = r"C:\Program Files\Gembird\Power Manager\pm.exe"):

        self.name ="USB connected PM (Windows)"
        self._socket = socket
        self._path = path
        self._on_string = "-on -PW1 -Scanner{0}".format(socket)
        self._off_string = "-off -PW1 -Scanner{0}".format(socket)

class LAN_PM(object):
    """Class for handling LAN-connected PM:s.

    host may be None if MAC is supplied. 
    If no password is supplied, default password is used."""

    def __init__(self, host, socket, password, verify_name = False, pm_name="Server 1", MAC=None):

        self.name ="LAN connected PM"
        self._host = host
        self._MAC = MAC
        self._socket = socket
        self._password = password is not None and password or "1"

        self._pm_server_name = pm_name
        self._pm_server_str = "<h2>{0}".format(pm_name)
        self._verify_name = verify_name

        self._pwd_params = urlencode((("pw", password),))
        self._on_params = urlencode((("cte{0}".format(socket),1),))
        self._on_params = urlencode((("cte{0}".format(socket),0),))
    

        self._login_out_url = "http://{0}/login.html".format(host)
        self._ctrl_panel_url = "http://{0}/".format(host)

        if MAC is not None:

            res = self._find_ip()

            if res is not None:
                self._host = res
    
    def _find_ip(self):
        """Looks up the MAC-address supplied on the local router"""

        #PINGSCAN ALL IP:S
        nm = nmap.PortScanner()
        nm_res = nm.scan(hosts="192.168.1.1-255", arguments="-sP")

        #FILTER OUT THOSE RESPONDING
        up_ips = [k for k in nm_res['scan'] if nm_res['scan'][k]['status']['state'] == u'up']

        #LET THE OS PING THEM AGAIN SO THEY END UP IN ARP
        for ip in up_ips:

            os.system('ping -c 1 {0}'.format(ip))

        #RUN ARP
        p = Popen(['arp','-n'], stdout=PIPE)

        #FILTER LIST ON ROWS WITH SOUGHT MAC-ADDRESS
        res = [l for l in p.communicate()[0].split("\n") if self._MAC in l]

        
        if len(res) > 0:
            #RETURN THE IP
            return res[0].split(" ",1)[0]

        else:
            #IF IT IS NOT CONNECTED AND UP RETURN NONE
            return None

    def _login(self):

        if self._host is None or self._host == "":

            return None
 
        else:

            return urllib2.urlopen(self._login_out_url, self._pwd_params)

    def _logout(self):

        if self._host is None or self._host == "":

            return None
 
        else:

            return urllib2.urlopen(self._login_out_url)

    def on(self):

        u = self._login()

        if u is None:

            return None

        if not self._verify_name or self._pm_server_str in u.readlines():

            urllib2.urlopen(self._ctrl_panel_url, self._on_params)

            self._logout()

    def off(self):

        u = self._login()

        if u is None:

            return None

        if not self._verify_name or self._pm_server_str in u.readlines():

            urllib2.urlopen(self._ctrl_panel_url, self._off_params)

            self._logout()

class Power_Manager():
    """Interface that takes a PM-class and emits messages as well as introduces
    a power toggle switch"""

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

