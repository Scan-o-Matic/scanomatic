#!/usr/bin/env python
"""Resource for scanning"""
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

from subprocess import Popen, PIPE
import re
import time
import copy
import uuid
import weakref

#
# INTERNAL DEPENDENCIES
#

import sane
import logger

#
# EXCEPTION
#


class Incompatible_Tuples(Exception):
    pass


class Failed_To_Claim_Scanner(Exception):
    pass


class Failed_To_Free_Scanner(Exception):
    pass


class Forbidden_Scanner_Owned_By_Other_Process(Exception):
    pass


class Unable_To_Write_To_Power_Up_Queue(Exception):
    pass


class Unable_To_Open(Exception):
    pass


class Corrupted_Lock_File(Exception):
    pass


class More_Than_One_Unknown_Scanner(Exception):
    pass


class Unknown_Scanner(Exception):
    pass


class No_Scanner(Exception):
    pass


class Risk_For_Orphaned_Scanner(Exception):
    pass


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

    USE_CALLBACK = False

    def __init__(self, parent, paths, config, name):

        self._logger = logger.Logger("Scanner {0}".format(name))
        self._parent = weakref.ref(parent) if parent else None
        self._paths = paths
        self._config = config
        self._name = name

        self._usb_address = None

        self._pm = self._config.get_pm(name)
        self._is_on = None

        self._lock_path = self._paths.lock_scanner_pattern.format(
            self._config.get_scanner_socket(self._name))

        self._lock_address_path = self._paths.lock_scanner_addresses

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

    def _free_in_file(self, soft):

        if self.get_claimed_by_other():

            if not soft:

                raise Forbidden_Scanner_Owned_By_Other_Process(
                    "Trying to free '{0}'".format(self._name))

            else:

                self._logger.info("Won't free scanner since owned by other")

            return False

        try:
            fs = open(self._lock_path, 'w')
            fs.close()
            self._logger.info("Scanner was freed in lock-file")
        except:
            raise Failed_To_Free_Scanner(self._name)

        self._remove_from_power_up_queue()
        if self._is_on is not False:
            self.off()

        return True

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

            self._logger.info("Scanner locked in file")
            return True

        else:
            self._logger.warning(
                "Scanner failed to lock in file since owned by other")

            return False

    def _queue_power_up_scan(self):

        self._queue_power_up_tries = 0

        while True:

            try:

                fs = open(self._paths.lock_power_up_new_scanner, 'a')
                fs.write("{0}\n".format(self._uuid))
                fs.close()
                self._logger.info("Queued for power up")
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

        self._logger.info("Waiting for power up turn")

        while cur_uuid != self._uuid:

            try:
                fs = open(self._paths.lock_power_up_new_scanner, 'r')
                cur_uuid = fs.readline().strip()
                fs.close()
            except:
                raise Unable_To_Open(self._paths.lock_power_up_new_scanner)
                return False

            if cur_uuid == "":
                raise Corrupted_Lock_File(self._paths.lock_power_up_new_scanner)
                return False

            time.sleep(1)

        self._logger.info("Scanner has power up rights")
        return True

    def _remove_from_power_up_queue(self):

        self._logger.info(
            "Scanner will be removed from power up queue")

        while True:

            try:
                fs = open(self._paths.lock_power_up_new_scanner, 'r')
                queue = fs.readlines()
                fs.close()
            except:
                raise Unable_To_Open(self._paths.lock_power_up_new_scanner)
                return False

            for i in range(len(queue) - 1, -1, -1):
                if queue[i].strip() == self._uuid:
                    del queue[i]

            try:

                fs = open(self._paths.lock_power_up_new_scanner, 'w')
                fs.writelines(queue)
                fs.close()
                self._logger.info("Scanner removed from queue")
                return True

            except:

                time.sleep(0.005)

        return False

    def _get_awake_scanners(self):

        p = Popen(
            "scanimage -L | " +
            r"""sed -n -E "s/^.*device[^\`]*.(.*libusb[^\`']*).*$/\1/p" """,
            #r"""sed -n -E "s/^.*found USB[^']*'(.*libusb[^']*).*$/\1/p" """,
            #" sed -n -E 's/^found USB.*(libusb.*$)/\\1/p'",
            shell=True, stdout=PIPE, stdin=PIPE)

        out, err = p.communicate()

        return [s for s in map(str, out.split('\n')) if len(s) > 0]

    def _get_scanner_address_lock(self):

        lock_states = dict()
        lines = list()
        try:
            fs = open(self._lock_address_path, 'r')
            lines = fs.readlines()
            fs.close()
        except:
            pass

        for line in lines:
            line_list = line.strip().split("\t")
            lock_states[line_list[0]] = line_list[1:]

        return lock_states

    def _write_scanner_address_claim(self):

        n_checks = 0
        while True:
            scanners = self._get_awake_scanners()
            lock_states = self._get_scanner_address_lock()
            free_scanners = [s for s in scanners
                             if s not in lock_states.keys()]

            if len(free_scanners) == 1:

                self._logger.info(
                    "Scanner located at address {0}".format(
                        free_scanners[0]))

                self._usb_address = free_scanners[0]

                try:
                    fs = open(self._lock_address_path, 'a')
                    fs.write("{0}\t{1}\n".format(free_scanners[0], self._name))
                    fs.close()
                except:
                    raise Unable_To_Open(self._lock_address_path)
                return True

            elif len(free_scanners) > 1:
                self._usb_address = None
                raise More_Than_One_Unknown_Scanner(free_scanners)

            else:
                self._usb_address = None

            n_checks += 1
            if n_checks > 10:
                raise No_Scanner()
            time.sleep(2)

    def _remove_scanner_address_lock(self):

        while True:

            s_list = list()
            my_addr = list()

            scanners = self._get_scanner_address_lock()
            for addr, s in scanners.items():
                if s is not None and len(s) > 0:
                    if s[0] != self._uuid:
                        s_list.append([addr] + s)
                    else:
                        my_addr.append(addr)

            awake_scanners = self._get_awake_scanners()
            awake_scanners = [s[0] for s in awake_scanners]
            s_list = [s for s in s_list if s in awake_scanners]
            my_addr = [a for a in my_addr if a not in awake_scanners]

            if len(my_addr) == 0:

                self._usb_address = None

                try:
                    fs = open(self._lock_address_path, 'w')
                    fs.writelines(["{0}\t{1}\n".format(*l) for l in s_list])
                    fs.close()
                    self._logger.info("No longer address locked")
                    break

                except:

                    self._logger.critical(
                        "I'm still on after when address lock removal")

                    raise Risk_For_Orphaned_Scanner(self._name)

            else:

                self._logger.critical(
                    "I'm still on after when address lock removal")

                raise Risk_For_Orphaned_Scanner(self._name)

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

    def get_power_status(self):

        return self._pm.status()

    def get_address(self):

        return self._usb_address

    """
            SETs
    """

    def set_uuid(self, s_uuid=None):

        if s_uuid is None:
            self._uuid = get_uuid()
        else:
            self._uuid = s_uuid

        self._logger.info("New uuid {0}".format(self._uuid))

    """
            ACTIONS
    """

    def claim(self):

        return self._lock_in_file()

    def on(self):

        is_on = None

        if self.get_claimed_by_other() is False:

            #Place self in queue to start scanner
            self._logger.debug("SCANNER, Queuing self for power up")
            self._queue_power_up_scan()

            #Wait for turn
            self._logger.debug("SCANNER, Waiting for turn to power up")
            self._wait_for_power_up_turn()

            #Power up and catch new scanner
            is_on = self._pm.on()
            if is_on:
                self._logger.info("Getting scanner address")
                self._write_scanner_address_claim()
            else:
                self._logger.error("SCANNER, Could not turn on!")

            #Allow next proc to power up scanner
            self._remove_from_power_up_queue()
            self._is_on = is_on

        return is_on

    def off(self, byforce=False):

        if self.get_claimed_by_other() is False or byforce:

            #Power down and remove scanner address lock
            is_on = not self._pm.off()
            self._remove_scanner_address_lock()
            self._is_on = is_on

        return not self._is_on

    def scan(self, mode, filename, auto_off=True):

        if self.get_claimed_by_other() is False:

            #Turn on
            if self._is_on is not True:
                self.on()

            #Scan
            if self._is_on:

                self._logger.info("Configurating for scan {0}".format(
                    self._parent().current_sane_settings))

                scanner = sane.Sane_Base(
                    owner=self,
                    model=self._model,
                    scan_mode=mode,
                    scan_settings=self._parent().current_sane_settings)

                scanner.AcquireByFile(filename=filename, scanner=self._usb_address)

                if auto_off:
                    self.off()

                return True

        else:

            raise Forbidden_Scanner_Owned_By_Other_Process(
                "Trying to scan on '{0}' with {1} while owned by {2}".format(
                self._name, self._uuid, self._get_check_in_file()))

        return False

    def free(self, soft=False):

        self._free_in_file(soft)
        return True


class Scanners(object):

    def __init__(self, paths, config):

        self._logger = logger.Logger("Scanners")
        self._paths = paths
        self._config = config
        self._scanners = dict()
        self._generic_naming = True
        self._sane_version = None
        self._sane_generic = \
            {"EPSON V700":
                {'TPU':
                    ["--source", "Transparency", "--format", "tiff",
                     "--resolution", "600", "--mode", "Gray", "-l", "0",
                     "-t", "0", "-x", "203.2", "-y", "254", "--depth", "8"],
                 'COLOR':
                    ["--source", "Flatbed", "--format", "tiff",
                     "--resolution", "300", "--mode", "Color", "-l", "0",
                     "-t", "0", "-x", "215.9", "-y", "297.18", "--depth", "8"]
                 }
             }

        #Highest version that takes a setting
        self._sane_flags_replace = {
            (1, 0, 22): [],
            (1, 0, 24): [(('EPSON V700', 'TPU', '--source'), 'TPU8x10')]
        }

        self._current_sane_settings = None

        self._set_sane_version()
        self.update()

    def __getitem__(self, key):

        return self._scanners[key]

    @property
    def current_sane_settings(self):
        return self._current_sane_settings

    def _get_sane_version(self):

        p = Popen([
            self._config.scan_program,
            self._config.scan_program_version_flag],
            shell=False, stdout=PIPE)

        stdout, stderr = p.communicate()

        self._backend_version = re.findall(r' ([0-9]+\.[0-9]+\.[0-9]+)',
                                           stdout.strip('\n'))

        return self._backend_version

    def _set_sane_version(self):

        #POS 0 is front-ends, 1 backends. Version is dot-serparated
        backend_version = map(int, self._get_sane_version()[1].split("."))

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
                    c_path[c_i + 1] = sane_value

        self._current_sane_settings = current

    def update_sane_setting(self, key, value, modes=['TPU', 'COLOR']):

        if self._current_sane_settings is not None:

            for model in self._current_sane_settings:

                for scan_mode in self._current_sane_settings[model]:

                    if (key in self._current_sane_settings[model][scan_mode]
                            and scan_mode in modes):

                        i = self._current_sane_settings[model][scan_mode].index(key)
                        if 0 <= i < len(self._current_sane_settings[model][scan_mode]) - 1:
                            self._current_sane_settings[model][scan_mode][i + 1] = value

                            self._logger.info(
                                "Updated scanning settings " +
                                "{2} for {0} in mode {1} to {3}".format(
                                    model, scan_mode, key, value))

    def update(self):

        scanner_count = self._config.number_of_scanners
        if self._generic_naming:
            scanner_name_pattern = self._config.scanner_name_pattern
            scanners = [scanner_name_pattern.format(s + 1) for s in xrange(scanner_count)]
        else:
            scanners = self._config.scanner_names

        for s in scanners:

            if s not in self._scanners.keys():
                self._scanners[s] = Scanner(self, self._paths, self._config, s)

        for s in self._scanners.keys():

            if s not in scanners:

                del self._scanners[s]

    def count(self, only_free=True):

        self.update()
        c = len([s for s in self._scanners.values()
                 if s.get_claimed() is False])

        return c

    def get_names(self, available=True):

        self.update()

        scanners = [s_name for s_name, s in self._scanners.items()
                    if available and
                    (s.get_claimed() is False or available is False)]

        return sorted(scanners)

    def claim(self, scanner_name):

        if scanner_name in self._scanners:
            return self._scanners[scanner_name].claim()
        else:
            raise Unknown_Scanner(scanner_name)
            return False

    def free(self, scanner_name, soft=False):
        if scanner_name in self._scanners:
            return self._scanners[scanner_name].free(soft=soft)
        else:
            raise Unknown_Scanner(scanner_name)
            return False
