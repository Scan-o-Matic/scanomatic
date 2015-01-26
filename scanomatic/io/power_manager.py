#!/usr/bin/env python
"""Resource module using a SIS-PM to control power to one socket at a time."""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os
import re
from subprocess import Popen, PIPE
import time

#LAN-specific dependencies
import urllib2
import types
from urllib import urlencode
from enum import Enum #NOTE: For python<3.4 use 'pip install enum34' to obtain module
#FURTHER LAN-specific dependenies further down

#
# INTERNAL DEPENDENCIES
#

import logger

#
# EXCEPTIONS
#


class Invalid_Init(Exception):
    pass

#
# GLOBALS
#

URL_TIMEOUT = 2
MAX_CONNECTION_TRIES = 10
POWER_MANAGER_TYPE = Enum("POWER_MANAGER_TYPE",
                     names=("notInstalled", "USB", "LAN", "linuxUSB", 
                            "windowsUSB"))
POWER_MODES = Enum("POWER_MODES", names=("Toggle", "Impulse"))
POWER_FLICKER_DELAY = 0.25

#
# FUNCTIONS
#


def _ImpulseScanner(self):
    onSuccess = self._on()
    time.sleep(POWER_FLICKER_DELAY)
    return self._off() and onSuccess


def _ToggleScannerOn(self):
    return self._on()


def _ToggleScannerOff(self):
    return self._off()


def hasValue(E, value):
    return any(elem for elem in E if elem.value == value)

def getEnumTypeFromValue(E, value):
    return list(elem for elem in E if elem.value == value)[0]

#
# CLASSES
#


class NO_PM(object):

    def __init__(self, socket, 
                 power_mode=POWER_MODES.Impulse, name="not installed"):

        if power_mode is POWER_MODES.Impulse:
            self.powerUpScanner = types.MethodType(_ImpulseScanner, self)
            self.powerDownScanner = types.MethodType(_ImpulseScanner, self)

        elif power_mode is POWER_MODES.Toggle:
            self.powerUpScanner = types.MethodType(_ToggleScannerOn, self)
            self.powerDownScanner = types.MethodType(_ToggleScannerOff, self)

        self._power_mode = power_mode
        self._socket = socket
        self.name = name
        self._logger = logger.Logger("Power Manager {0}".format(name))

    @property
    def power_mode(self):
        return str(self._power_mode)

    def _on(self):


        return True

    def _off(self):

        return True

    def status(self):

        self._logger.warning("claiming to be off")
        return False

    def couldHavePower(self):

        return (self._power_mode is POWER_MODES.Impulse or 
                self.status() is not False)

    def sureToHavePower(self):

        return (self._power_mode is not POWER_MODES.Impulse and
                self.status() is not False)


class USB_PM(NO_PM):
    """Base Class for USB-connected PM:s. Not intended to be used directly."""
    def __init__(self, socket, path, on_args=[], off_args=[],
                 power_mode=POWER_MODES.Impulse, name="USB"):

        super(USB_PM, self).__init__(socket, power_mode=power_mode, name=name)

        self._on_cmd = [path] + on_args
        self._off_cmd = [path] + off_args
        self._fail_error = "No GEMBIRD SiS-PM found"

    def _on(self):
        on_success = self._exec(self._on_cmd)
        self._logger.info('USB PM, Turning on socket {0} ({1})'.format(
            self._socket, on_success))
        return on_success

    def _off(self):
        off_success = self._exec(self._off_cmd)
        self._logger.info('USB PM, Turning off socket {0} ({1})'.format(
            self._socket, off_success))
        return off_success

    def _exec(self, cmd):

        exec_err = False
        stderr = ""
        try:
            proc = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)
            stdout, stderr = proc.communicate()
        except:
            exec_err = True
        if self._fail_error in stderr or exec_err:
            return False

        return True


class USB_PM_LINUX(USB_PM):
    """Class for handling USB connected PM:s on linux."""

    ON_TEXT = "on"
    OFF_TEXT = "off"

    def __init__(self, socket, path="sispmctl",
                 power_mode=POWER_MODES.Impulse):

        super(USB_PM_LINUX, self).__init__(
            socket,
            path,
            on_args=["-o", "{0}".format(socket)],
            off_args=["-f", "{0}".format(socket)],
            power_mode=power_mode,
            name="USB(Linux)")

    def status(self):

        self._logger.info('USB PM, trying to connect')
        proc = Popen('sispmctl -g {0}'.format(self._socket),
                     stdout=PIPE, stderr=PIPE, shell=True)

        stdout, stderr = proc.communicate()
        stdout = stdout.strip()

        if stdout.endswith(self.ON_TEXT):
            return True
        elif stdout.endswith(self.OFF_TEXT):
            return False

        self._logger.warning('USB PM, could not reach or understand PM')

        return None


class USB_PM_WIN(USB_PM):
    """Class for handling USB connected PM:s on windows."""
    def __init__(self, socket,
                 path=r"C:\Program Files\Gembird\Power Manager\pm.exe",
                 power_mode=POWER_MODES.Impulse):

        super(USB_PM_LINUX, self).__init__(
            socket,
            path,
            on_args=["-on", "-PW1", "-Scanner{0}".format(socket)],
            off_args=["-off", "-PW1", "-Scanner{0}".format(socket)],
            power_mode=power_mode,
            name="USB(Windows)")


class LAN_PM(NO_PM):
    """Class for handling LAN-connected PM:s.

    host may be None if MAC is supplied.
    If no password is supplied, default password is used."""

    def __init__(self, socket, host=None, password="1", verify_name=False,
                 pm_name="Server 1", MAC=None,
                 power_mode=POWER_MODES.Impulse):

        super(LAN_PM, self).__init__(socket, name="LAN",
                                     power_mode=power_mode)
        self._host = host
        self._MAC = MAC
        if password is None:
            password = "1"
        self._password = password

        self._pm_server_name = pm_name
        self._pm_server_str = "<h2>{0}".format(pm_name)
        self._verify_name = verify_name

        self._pwd_params = urlencode((("pw", password),))
        self._on_params = urlencode((("cte{0}".format(socket), 1),))
        self._off_params = urlencode((("cte{0}".format(socket), 0),))

        self._set_urls()
        self.test_ip()

        if self._host is None:

            if  MAC is not None:

                self._logger.info("LAN PM, No valid host known, searching...")
                res = self._find_ip()
                self._logger.info("LAN PM, Found {0}".format(res))

            else:

                self._logger.error("LAN PM, No known host and no MAC...no way to find PM")
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
        p = Popen(['arp', '-n'], stdout=PIPE)

        #FILTER LIST ON ROWS WITH SOUGHT MAC-ADDRESS
        self._logger.debug("LAN PM, Keeping those with correct MAC-addr")

        res = [l for l in p.communicate()[0].split("\n") if self._MAC in l]

        if len(res) > 0:
            #RETURN THE IP

            for r in res:
                self._host = r.split(" ", 1)[0]
                self._set_urls()
                if self.test_ip() is not None:
                    break
        else:
            #IF IT IS NOT CONNECTED AND UP RETURN NONE
            self._host = None

        self._set_urls()

        return self._host

    def _run_url(self, *args, **kwargs):

        success = False
        connects = 0
        p = None

        while not success and connects < MAX_CONNECTION_TRIES:

            try:
                p = urllib2.urlopen(*args, **kwargs)
                success = True
            except:
                connects += 1

        if connects == MAX_CONNECTION_TRIES:
            self._logger.error("Failed to reach PM ({0} tries)".format(connects))

        return p

    def _login(self):

        if self._host is None or self._host == "":

            self._logger.error("LAN PM, Loging in failed, no host")
            return None

        else:

            self._logger.info("LAN PM, Logging in")
            return self._run_url(self._login_out_url, self._pwd_params, timeout=URL_TIMEOUT)

    def _logout(self):

        if self._host is None or self._host == "":

            self._logger.error("LAN PM, Log out failed, no host")
            return None

        else:

            self._logger.info("LAN PM, Logging out")
            return self._run_url(self._login_out_url, timeout=URL_TIMEOUT)

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

    def _on(self):

        u = self._login()

        if u is None:

            return False

        if not self._verify_name or self._pm_server_str in u.read():

            self._logger.info('USB PM, Turning on socket {0}'.format(self._socket))
            if self._run_url(self._ctrl_panel_url, self._on_params, timeout=URL_TIMEOUT) is None:
                return False

            self._logout()
            return True

        else:

            self._logger.error("LAN PM, Failed to turn on socket {0}".format(self._socket))
            return False

    def _off(self):

        u = self._login()

        if u is None:

            return False

        if not self._verify_name or self._pm_server_str in u.read():

            self._logger.info('USB PM, Turning off socket {0}'.format(self._socket))

            if self._run_url(self._ctrl_panel_url, self._off_params,
                             timeout=URL_TIMEOUT) is None:

                return False

            self._logout()
            return True

        else:

            self._logger.error("LAN PM, Failed to turn off socked {0}".format(self._socket))
            return False

    def status(self):

        u = self._login()

        if u is None:
            self._logger.error('Could not reach LAN-PM')
            return None

        page = u.read()
        if not self._verify_name or self._pm_server_str in page:

            states = re.findall(r'sockstates = ([^;]*)', page)[0].strip()
            try:
                states = eval(states)
                if len(states) >= self._socket:
                    return states[self._socket - 1] == 1
            except:
                pass

        return None


class Power_Manager():
    """Interface that takes a PM-class and emits messages as well as introduces
    a power toggle switch"""

    def __init__(self, pm=None):

        if pm is None:
            pm = NO_PM()
        self._pm = pm
        self._installed = pm is not None
        self._on = None
        self._logger = logger.Logger("Power Manager")

        if pm is not None:
            self._logger.debug("Power Hooked up {0}, Socket {1}".format(
                pm.name,
                pm._socket))

        else:
            self._logger.debug("No device to talk to")

    def _on(self):

        if self._installed and self._on is not True:
            self._pm.powerUpScanner()
            self._on = True
            self._logger.debug("Switching on {0}, Socket {1}".format(
                self._pm.name, self._pm._socket))

    def _off(self):
        if self._installed and self._on is not False:
            self._pm.powerDownScanner()
            self._on = False
            self._logger.debug("Switching off {0}, Socket {1}".format(
                self._pm.name, self._pm._socket))

    def status(self):
        print "Power Manager, checking status"
        return self._pm.status()

    def toggle(self):
        if self.on is None or self.on is False:
            self.on()
        else:
            self.off()
