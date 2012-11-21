#!/usr/bin/env python
"""Resource scanner """
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

from subprocess import Popen, PIPE
import re
import time
import copy
import uuid

#
# INTERNAL DEPENDENCIES
#

import src.resource_sane as resource_sane
import src.resource_logger as resource_logger

#
# EXCEPTION
#

class Incompatible_Tuples(Exception): pass
class Failed_To_Claim_Scanner(Exception): pass
class Forbidden_Scanner_Owned_By_Other_Process(Exception): pass
class Unable_To_Open(Exception): pass
class Corrupted_Lock_File(Exception): pass
class More_Than_One_Unknown_Scanner(Exception): pass
class Unknown_Scanner(Exception): pass


#
# FUNCTIONS
#

def version_lq(a, b):
    """Compares a version tuple/list a to b by ordered elements.
    If a <= b returns True, else False"""

    if len(a) != len(b):
        raise Incompatible_Tuples("{0} {1}".format(a, b))

    for i in range(len(a)):
        if a[i] > b[i]:
            return False

    return True


def get_uuid():

    return uuid.uuid1().get_urn().split(":")[-1]

#
# CLASSES
#


class Scanner(object):

    _USE_CALLBACK = False

    def __init__(self, parent, paths, config, name, logger):

        self._logger = logger
        self._parent = parent
        self._paths = paths
        self._config = config
        self._name = name

        self._pm = self._config.get_pm(name, logger=logger)

        self._lock_path = self._paths.lock_scanner_pattern.format(
            self._config.get_scanner_socket(self._name))
        self._queue_power_up_tries = 0

        self._uuid = get_uuid()

        self._model = self._config.get_scanner_model(self._name)

    """
            LOCK FILES
    """
    def _get_check_in_file(self):

        lock_uuid = ""

        try:
            fs = open(self._lock_path, 'r')
            lock_uuid = fs.read().strip()
            fs.close()
        except:
            pass

        return lock_uuid

    def _free_in_file(self):

        if self.get_claimed_by_other():
            raise Forbidden_Scanner_Owned_By_Other_Process(
                "Trying to free '{0}'".format(self._name))
            return

        try:
            fs = open(self._lock_path, 'w')
            fs.close()
        except:
            Failed_To_Free_Scanner(self._name)

    def _lock_in_file(self):

        lock_uuid = self._get_check_in_file()

        if lock_uuid == "" or lock_uuid == self._uuid:

            if lock_uuid == "":

                try:
                    fs = open(self._lock_path, 'w')
                    fs.write(self._uuid)
                    fs.close()
                except:
                    raise Failed_To_Claim_Scanner(self._name)

            return True

        else:

            return False

    def _queue_power_up_scan(self):

        self._queue_power_up_tries = 0

        while True:

            try:
                fs = open(self._paths.lock_power_up_new_scanner, 'a')
                fs.write("{0}\n".format(self._uuid))
                fs.close()
                break
            except:
                self._queue_power_up_tries += 1
                if self._queue_power_up_tries >= 40:
                    raise Unable_To_Write_To_Power_Up_Queue(
                        "Tried {0} times".format(self._queue_power_up_tries))

                    return False

                time.sleep(0.005)

        return True

    def _wait_for_power_up_turn(self):

        cur_uuid = None 

        while cur_uuid != self._uuid:

            try:
                fs = open(self._paths.lock_power_up_new_scanner, 'r')
                cur_uuid = fs.readline().strip()
                fs.close()
            except:
                raise Unable_To_Open(self._paths.lock_power_up_new_scanner)

            if cur_uuid == "":
                raise Corrupted_Lock_File(self._paths.lock_power_up_new_scanner)

            time.sleep(1)

    def _remove_from_power_up_queue(self):

        try:
            fs = open(self._paths.lock_power_up_new_scanner, 'r')
            queue = fs.readlines()
            fs.close()
        except:
            raise Unable_To_Open(self._paths.lock_power_up_new_scanner)

        if queue[0].strip() != self._uuid:
            return False

        while True:

            try:
                fs = open(self._paths.lock_power_up_new_scanner, 'w')
                fs.writelines(queue[1:])
                fs.close()
                return True

            except:

                time.sleep(0.005)

    def _get_awake_scanners(self):

        p = subprocess.Popen("sane-find-scanner -v -v |" +
            " sed -n -E 's/^found USB.*(libusb.*$)/\\1/p'",
            shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        out, err = p.communicate()

        return map(str, out.split('\n'))

    def _get_scanner_address_lock(self):

        lock_states = dict()
        try:
            fs = open(self._lock_path, 'r')
            lines = fs.readlines()
            fs.close()
        except:
            raise Unable_To_Open(self._lock_path)

        for line in lines:
            line_list = line.strip().split("\t")
            lock_states[line_list[0]] = line_list[1:]
    
    def _write_scanner_address_claim(self):

        while True:
            scanners = self._get_awake_scanners()
            lock_states = self._get_scanner_address_lock()
            free_scanners = [s for s in scanners if s not in lock_states.keys()]

            if len(free_scanners) == 1:

                try:
                    fs = open(self._lock_path, 'a')
                    fs.write("{0}\t{1}\n".format(free_scanners[0], self._name))
                    fs.close()
                except:
                    raise Unable_To_Open(self._lock_path)
                return True

            elif len(free_scanners):
                raise More_Than_One_Unknown_Scanner(free_scanners)

            time.sleep(2)

    def _remove_scanner_address_lock(self):

        while True:

            s_list = list()
            my_addr = list()

            scanners = self._get_scanner_address_lock()
            for addr, s in scanners.items():
                if s[0] != self._uuid:
                    s_list.append([addr]+s)
                else:
                    my_addr.append(addr)

            awake_scanners = self._get_awake_scanners()
            awake_scanners = [s[0] for s in awake_scanner]

            my_addr = [a for a in my_addr if a not in awake_scanners]
            
            if len(my_addr) == 0:
                try:
                    fs = open(self._lock_path, 'w')
                    fs.writelines(["{0}\t{1}\n".format(*l) for l in s_list])
                    fs.close()
                    break
                except:
                    pass

            time.sleep(0.005)

    """
            GETs
    """

    def get_claimed(self):

        return self._get_check_in_file() != "" 

    def get_claimed_by_other(self):
        """True if owned by other, else False"""
        owner_uuid = self._get_check_in_file()
        return not(owner_uuid == "" or owner_uuid == self._uuid)

    def get_name(self):

        return self._name

    def get_uuid(self):

        return self._uuid

    def get_socket(self):

        return self._config.get_scanner_socket(self._name)

    """
            SETs
    """

    def set_uuid(self, s_uuid):

        self._uuid = s_uuid

    """
            ACTIONS
    """
    def claim(self):

        return self._lock_in_file()

    def scan(self, mode, filename):

        if self.get_claimed_by_other() == False:

            #Place self in queue to start scanner
            self._logger.debug("SCANNER, Queuing self for power up")
            self._queue_power_up_scan()

            #Wait for turn
            self._logger.debug("SCANNER, Waiting for turn to power up")
            self._wait_for_power_up_turn()

            #Power up and catch new scanner
            is_on = self._pm.on()
            if is_on:
                self._get_scanner_address_lock()
            else:
                self._logger.error("SCANNER, Could not turn on!")

            #Allow next proc to power up scanner
            self._remove_from_power_up_queue()

            #Scan
            if is_on:
                scanner = resource_sane.Sane_Base(owner=self,
                    model=_model,
                    scan_mode=mode,
                    output_function=self._logger,
                    scan_settings=self._parent._current_sane_setting)

                scanner.AcquireByFile(filename=filename)

                #Power down and remove scanner address lock
                self._pm.off()
                self._remove_scanner_address_lock()

                return True

        else:

            raise Forbidden_Scanner_Owned_By_Other_Process(
                "Trying to scan on '{0}' with {1} while owned by {2}".format(
                self._name, self._uuid, self._get_check_in_file()))

        return False

    def free(self):

        self._free_in_file()
        return True


class Scanners(object):

    def __init__(self, paths, config, logger=None):

        if logger is not None:
            self._logger = logger
        else:
            self._logger = resource_logger.Fallback_Logger()

        self._paths = paths
        self._config = config
        self._scanners = dict()
        self._generic_naming = True
        self._sane_version = None
        self._sane_generic = \
            {"EPSON V700" :
                {'TPU': 
                    ["--source", "Transparency" ,"--format", "tiff", 
                    "--resolution", "600", "--mode", "Gray", "-l", "0",  
                    "-t", "0", "-x", "203.2", "-y", "254", "--depth", "8"],
                'COLOR':
                    ["--source", "Flatbed", "--format", "tiff",
                    "--resolution", "300", "--mode", "Color", "-l", "0",
                    "-t", "0", "-x", "215.9", "-y", "297.18", "--depth", "8"]} }

        #Highest version that takes a setting
        self._sane_flags_replace = {
            (1,0,22): [], 
            (1,0,24): [(('EPSON V700', 'TPU', '--source'), 'TPU8x10')] }

        self._current_sane_settings = None

        self._set_sane_version()
        self.update()

    def __getitem__(self, key):

        return self._scanners[key]

    def _get_sane_version(self):

        p = Popen([self._config.scan_program, 
            self._config.scan_program_version_flag],
            shell=False, stdout=PIPE)

        stdout, stderr = p.communicate()

        self._backend_version = re.findall(r' ([0-9]+\.[0-9]+\.[0-9]+)', 
            stdout.strip('\n'))

        return self._backend_version

    def _set_sane_version(self):

        #POS 0 is front-ends, 1 backends. Version is dot-serparated
        backend_version = map(int , self._get_sane_version()[1].split("."))

        for v in sorted(self._sane_flags_replace.keys()):

            if version_lq(backend_version, v):

                self._sane_version = v
                break

        if self._sane_version is None:
            
            print "***WARNING:\tHoping driver works as last known version"
            self._sane_versi0n = v

        current = copy.deepcopy(self._sane_generic)

        #Changes values in the call tuple according to what is needed
        #for the specific version
        for sane_path, sane_value in self._sane_flags_replace[v]:

            c_path = current
            for s_p in sane_path[:-1]:
                c_path = c_path[s_p]

            for c_i, c_v in enumerate(c_path[:-1]):
                if c_v == sane_path[-1]:
                    c_path[c_i] = sane_value


        self._current_sane_settings = current


    def update(self):

        scanner_count = self._config.number_of_scanners
        if self._generic_naming:
            scanner_name_pattern = "Scanner {0}"
            scanners = [scanner_name_pattern.format(s + 1) for s in xrange(scanner_count)]
        else:
            scanners = self._config.scanner_names

        for s in scanners:

            if s not in self._scanners.keys():
                self._scanners[s] = Scanner(self, self._paths, self._config, s, self._logger)

        for s in self._scanners.keys():

            if s not in scanners:

                del self._scanners[s]

    def count(self, only_free=True):

        self.update()
        c = len([s for s in self._scanners.values() 
                    if s.get_claimed() == False])

        return c

    def names(self, available=True):

        self.update()

        scanners = [s_name for s_name, s in self._scanners.items() if available and \
            s.get_claimed() == False or available == False]

        return sorted(scanners)

    def claim(self, scanner_name):

        if scanner_name in self._scanners:
            return self._scanners[scanner_name].claim()
        else:
            raise Unknown_Scanner(scanner_name)
            return False

    def free(self, scanner_name):
        if scanner_name in self._scanners:
            return self._scanners[scanner_name].free()
        else:
            raise Unknown_Scanner(scanner_name)
            return False


