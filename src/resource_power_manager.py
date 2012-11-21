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
from subprocess import Popen, PIPE

#LAN-specific dependencies
import urllib2
from urllib import urlencode
#FURTHER LAN-specific dependenies further down

#
# INTERNAL DEPENDENCIES
#

import src.resource_logger as resource_logger

#
# EXCEPTIONS
#

class Invalid_Init(Exception): pass

#
# CLASSES
#

class USB_PM(object):
    """Base Class for USB-connected PM:s. Not intended to be used directly."""
    def __init__(self, path, on_args=[], off_args=[], logger=None):

        self.name ="USB connected PM"
        self._on_cmd = [path] + on_args
        self._off_cmd = [path] + off_args
        self._fail_error = "No GEMBIRD SiS-PM found"

        if logger is None:
            self._logger = logger
        else:
            self._logger = resource_logger.Log_Garbage_Collector()

    def on(self):
        return self._exec(self._on_cmd)

    def off(self):
        return self._exec(self._off_cmd)

    def _exec(self, cmd):
        proc = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        if self._fail_error in stderr:
            return False

        return True



class USB_PM_LINUX(USB_PM):
    """Class for handling USB connected PM:s on linux."""
    def __init__(self, socket, path = "sispmctl", logger=None):

        super(USB_PM_LINUX, self).__init__(path,
            on_args=["-o", "{0}".format(socket)],
            off_args=["-f", "{0}".format(socket)],
            logger=logger)

        self.name ="USB connected PM (Linux)"
        self._socket = socket

class USB_PM_WIN(USB_PM):
    """Class for handling USB connected PM:s on windows."""
    def __init__(self, socket, path = r"C:\Program Files\Gembird\Power Manager\pm.exe",
            logger=None):

        super(USB_PM_LINUX, self).__init__(path,
            on_args=["-on", "-PW1", "-Scanner{0}".format(socket)],
            off_args=["-off", "-PW1", "-Scanner{0}".format(socket)],
            logger=logger)

        self.name ="USB connected PM (Windows)"
        self._socket = socket

class LAN_PM(object):
    """Class for handling LAN-connected PM:s.

    host may be None if MAC is supplied. 
    If no password is supplied, default password is used."""

    def __init__(self, socket, host=None, password=None, verify_name=False,
            pm_name="Server 1", MAC=None, logger=None):

        self.name ="LAN connected PM"
        self._host = host
        self._MAC = MAC
        self._socket = socket
        self._password = password is not None and password or "1"

        if logger is None:
            self._logger = resource_logger.Log_Garbage_Collector()
        else:
            self._logger = logger

        self._pm_server_name = pm_name
        self._pm_server_str = "<h2>{0}".format(pm_name)
        self._verify_name = verify_name

        self._pwd_params = urlencode((("pw", password),))
        self._on_params = urlencode((("cte{0}".format(socket),1),))
        self._off_params = urlencode((("cte{0}".format(socket),0),))

        self._set_urls()
        self.test_ip()

        if self._host is None:

            if  MAC is not None:

                self._logger.info("LAN PM, No valid host known, searching...")
                res = self._find_ip()
                self._logger.info("LAN PM, Found {0}".format(res))

            else:

                self._logger.error("LAN PM, No knowon host and no MAC...no way to find PM")
                raise Invalid_Init()
    
    def _set_urls(self):

        host = self._host

        self._login_out_url = "http://{0}/login.html".format(host)
        self._ctrl_panel_url = "http://{0}/".format(host)

    def _find_ip(self):
        """Looks up the MAC-address supplied on the local router"""

        #SEARCHING FOR IP SPECIFIC DEPENDENCIES
        import nmap

        #PINGSCAN ALL IP:S
        self._logger.info("LAN PM, Scanning hosts (may take a while...)")
        nm = nmap.PortScanner()
        nm_res = nm.scan(hosts="192.168.0.1-255", arguments="-sP")


        #FILTER OUT THOSE RESPONDING
        self._logger.debug("LAN PM, Evaluating all alive hosts")
        up_ips = [k for k in nm_res['scan'] if nm_res['scan'][k]['status']['state'] == u'up']

        #LET THE OS PING THEM AGAIN SO THEY END UP IN ARP
        self._logger.debug("LAN PM, Scanning pinning alive hosts")
        for ip in up_ips:

            os.system('ping -c 1 {0}'.format(ip))

        #RUN ARP
        self._logger.debug("LAN PM, Searching arp")
        p = Popen(['arp','-n'], stdout=PIPE)

        #FILTER LIST ON ROWS WITH SOUGHT MAC-ADDRESS
        self._logger.debug("LAN PM, Keeping those with correct MAC-addr")

        res = [l for l in p.communicate()[0].split("\n") if self._MAC in l]

        
        if len(res) > 0:
            #RETURN THE IP

            for r in res:
                self._host = r.split(" ",1)[0]
                self._set_urls()
                if self.test_ip() is not None:
                    break
        else:
            #IF IT IS NOT CONNECTED AND UP RETURN NONE
            self._host = None

        self._set_urls()

        return self._host

    def _login(self):

        if self._host is None or self._host == "":

            self._logger.error("LAN PM, Loging in failed, no host")
            return None
 
        else:

            self._logger.info("LAN PM, Logging in")
            return urllib2.urlopen(self._login_out_url, self._pwd_params)

    def _logout(self):

        if self._host is None or self._host == "":

            self._logger.error("LAN PM, Log out failed, no host")
            return None
 
        else:

            self._logger.info("LAN PM, Logging out")
            return urllib2.urlopen(self._login_out_url)

    def test_ip(self):

        self._logger.debug("LAN PM, Testing current host '{0}'".format(self._host))

        if self._host is not None:

            u = self._logout()

            if u is None:

                self._host = None

            else:
            
                s = u.read()
                u.close()

                if "EnerGenie" not in s:

                    self._host = None

                if self._pm_server_name not in s:

                    self._host = None

        return self._host is not None
            
    def on(self):

        u = self._login()

        if u is None:

            return False

        if not self._verify_name or self._pm_server_str in u.read():

            self._logger.info("LAN PM, Turning on")
            urllib2.urlopen(self._ctrl_panel_url, self._on_params)

            self._logout()
            return True

        else:

            self._logger.error("LAN PM, Failed to turn on")
            return False

    def off(self):

        u = self._login()

        if u is None:

            return False

        if not self._verify_name or self._pm_server_str in u.read():

            self._logger.info("LAN PM, Turning off")
            urllib2.urlopen(self._ctrl_panel_url, self._off_params)

            self._logout()
            return True

        else:

            self._logger.error("LAN PM, Failed to turn off")
            return False

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
            self._DMS = resource_logger.Log_Garbage_Collector()

        if pm is not None:
            self._DMS("Power", "Hooked up {0}, Socket {1}".format(pm.name, 
                pm._socket))

            self._pm.set_DMS(DMS)

        else:
            self._DMS("Power", "Power Manager has no device to talk to")

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

