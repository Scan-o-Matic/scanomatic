#!/usr/bin/env python
"""Resource for managing turning scanners on and off"""
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
import psutil

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
import scanomatic.io.fixtures as fixtures
from scanomatic.models.factories.scanning_factory import ScannerOwnerFactory
from scanomatic.io.power_manager import InvalidInit, PowerManagerNull
import scanomatic.generics.decorators as decorators
from scanomatic.generics.singleton import Singleton


def get_alive_scanners():

    p = Popen(["scanimage", "-L"], shell=False, stdout=PIPE, stderr=PIPE)
    stdout, _ = p.communicate()
    return re.findall(r'device[^\`]*.(.*libusb[^\`\']*)', stdout)


JOB_CALL_SCANNER_REQUEST_ON = "request_scanner_on"
JOB_CALL_SCANNER_REQUEST_OFF = "request_scanner_off"

class ScannerPowerManager(Singleton):

    def __init__(self):

        self._logger = logger.Logger("Scanner Manager")
        self._conf = app_config.Config()
        self._paths = paths.Paths()
        self._fixtures = fixtures.Fixtures()
        self._orphan_usbs = set()

        self._scanners = self._get_scanner_owners_from_file()
        self._pm = self._get_power_manager(self._scanners)
        self._scanner_queue = []
        decorators.register_type_lock(self)

    def __getitem__(self, item):

        """

        :rtype : scanomatic.models.scanning_model.ScannerOwnerModel
        """
        if isinstance(item, int):
            return self._scanners[item]
        return [scanner for scanner in self._scanners.values() if scanner.scanner_name == item][0]

    def __contains__(self, item):

        """

        :rtype : bool
        """
        try:
            return self[item] is not None
        except IndexError:
            return False

    def _get_scanner_owners_from_file(self):

        scanners = {}

        for scanner in ScannerOwnerFactory.serializer.load(self._paths.config_scanners):

            scanners[scanner.socket] = scanner

        for socket in self._enumerate_scanner_sockets():
            if socket not in scanners:
                scanner = ScannerOwnerFactory.create(socket=socket, scanner_name=self._conf.get_scanner_name(socket))
                scanners[scanner.socket] = scanner

        self._logger.info("Scanners inited: {0}".format(scanners))

        return scanners

    def _enumerate_scanner_sockets(self):

        for power_socket in range(self._conf.number_of_scanners):
            yield power_socket

    def _get_power_manager(self, scanners):

        pm = {}
        for power_socket in scanners:

            try:
                pm[power_socket] = self._conf.get_pm(power_socket)
            except InvalidInit:
                self._logger.error("Failed to init socket {0}".format(power_socket))
                pm[power_socket] = PowerManagerNull(power_socket)

        self._logger.info("Power Managers inited {0}".format(pm))
        return pm

    def _save(self, scanner_owner_model):

        ScannerOwnerFactory.serializer.dump(scanner_owner_model, self._paths.config_scanners)

    def _rescue(self, available_usbs, active_usbs):

        self._orphan_usbs = self._orphan_usbs.union(available_usbs)

        power_statuses = self.power_statuses
    
        for scanner in self._scanners.values():

            could_have_or_claims_to_have_power = power_statuses[scanner.socket] or scanner.power
            no_or_bad_usb = not scanner.usb or scanner.usb not in active_usbs

            if could_have_or_claims_to_have_power and no_or_bad_usb:

                if self._power_down(scanner):

                    self._save(scanner)

    def _match_scanners(self, active_usbs):

        self._trim_no_longer_active_orphan_uabs(active_usbs)
        available_usbs = self._get_non_orphan_usbs(active_usbs)
        unknown_usbs = self._remove_known_usbs(available_usbs)

        if not unknown_usbs:
            return True

        if not self._can_assign_usb(unknown_usbs):
            self._rescue(unknown_usbs, active_usbs)
            return False

        self._assign_usb_to_claim(unknown_usbs)

        return True

    @property
    def _claimer(self):

        for scanner in self._scanners.values():
            if scanner.claiming:
                return scanner
        return None

    def _can_assign_usb(self, unknown_usbs):

        if len(unknown_usbs) > 1 or not self._claimer:
            self._logger.critical("More than one unclaimed scanner {0}".format(
                unknown_usbs))
            return False
        return True

    def _assign_usb_to_claim(self, unknown_usbs):

        scanner = self._claimer
        scanner.usb = unknown_usbs[0]
        scanner.claiming = False
        self._save(scanner)

    def _set_usb_to_scanner_that_could_be_on(self, usb):

        if self._claimer:
            return False

        powers = self.power_statuses
        if sum(powers.values()) == 1:

            socket = tuple(scanner for scanner in powers if powers[scanner])[0]
            scanner = self._scanners[socket]

            if self._pm[socket].sure_to_have_power():
                scanner.power = True
                scanner.usb = usb
                self._save(scanner)
                return True
            else:
                self._logger.critical(
                    "There's one scanner on {0}, but can't safely assign it to {1}".format(
                        usb, scanner))
                return False
        else:
            self._logger.critical(
                "There's one scanner on {0}, but non that claims to be".format(
                    usb))
            return False
                    
    def _remove_known_usbs(self, available_usbs):

        known_usbs = set(scanner.usb for scanner in self._scanners.values() if scanner.usb)
        return set(usb for usb in available_usbs if usb not in known_usbs)

    def _trim_no_longer_active_orphan_uabs(self, active_usbs):

        self._orphan_usbs = self._orphan_usbs.intersection(active_usbs)

    def _get_non_orphan_usbs(self, usbs):

        return set(usbs).difference(self._orphan_usbs)

    def owns_scanner(self, owner_pid, job_id):

        return any(True for scanner in self._scanners.values()
                   if scanner.job_id == job_id and scanner.owner_pid == owner_pid)

    def request_on(self, job_id):

        scanner = self._get_scanner_by_job_id(job_id)
        if scanner:
            if scanner.usb:
                return scanner.usb
            else:
                return self._add_to_claim_queue(scanner)

        else:
            self._logger.warning("No scanner has been claimed by {0}".format(job_id))
            return False

    @decorators.type_lock
    def _add_to_claim_queue(self, scanner):

        if scanner not in self._scanner_queue:
            self._logger.info("Added scanner to queue for on/off action")
            self._scanner_queue.append(scanner)
        return True

    def request_off(self, job_id):

        scanner = self._get_scanner_by_job_id(job_id)

        if not scanner:
            self._logger.error(
                "Can't turn off scanner for unknown job {1}".format(job_id))
            return False

        if self._power_down(scanner):
            self._save(scanner)
            return True
        return False

    def _power_down(self, scanner):

        success = self._pm[scanner.socket].powerDownScanner()

        if success:
            scanner.usb = ""
            scanner.power = False

        return success

    def request_claim(self, rpc_job_model):

        scanner = rpc_job_model.contentModel.scanner
        scanner_name = self._conf.get_scanner_name(scanner)

        if scanner not in self._scanners:
            self._logger.warning("Unknown scanner referenced ({0})".format(
                scanner_name))
            return False

        scanner_owner_model = self._scanners[scanner]

        if scanner_owner_model.job_id != rpc_job_model.id:

            if psutil.pid_exists(scanner_owner_model.owner_pid):

                self._logger.warning("Trying to claim {0} when claimed".format(
                    scanner_name))
                return False

            else:
                self._logger.info(
                    "Releasing {0} since owner process is dead".format(
                        scanner_name))

                scanner_owner_model.job_id = ""
                self._power_down(scanner_owner_model)

        scanner_owner_model.job_id = rpc_job_model.id
        scanner_owner_model.owner_pid = rpc_job_model.pid

        self._save(scanner_owner_model)

        return True

    def release_scanner(self, job_id):

        scanner = self._get_scanner_by_job_id(job_id)
        if scanner.power or scanner.usb:
            self._power_down(scanner)

        scanner.owner = ""
        self._save(scanner)

        return True

    def _get_scanner_by_job_id(self, job_id):

        scanners = [scanner for scanner in self._scanners.values() if scanner.job_id == job_id]
        if scanners:
            return scanners[0]
        return None

    def update(self):

        self._manage_claimer()
        return self._match_scanners(get_alive_scanners())

    @decorators.type_lock
    def _manage_claimer(self):

        if not self._claimer and self._scanner_queue:
            scanner = self._scanner_queue.pop(0)
            while scanner in self._scanner_queue:
                self._scanner_queue.remove(scanner)
            scanner.claiming = True
            self._save(scanner)

        claimer = self._claimer
        if claimer and not claimer.power:

            claimer.power = self._pm[self._claimer.socket].powerUpScanner()
            self._save(claimer)

    def has_fixture(self, fixture_name):

        return fixture_name in self._fixtures

    @property
    def status(self):
        return self._scanners.values()

    @property
    def fixtures(self):
        return self._fixtures.get_names()

    @property
    def power_statuses(self):
        return {scanner_socket: self._pm[scanner_socket].could_have_power()
                for scanner_socket in self._pm}

    @property
    def has_scanners(self):
        reachable_pms = any(type(pm) is not PowerManagerNull for pm in self._pm.values())
        self._logger.info("Power Manager {0} is reachable? {1}".format(self._pm, reachable_pms))
        return self._pm and reachable_pms

    @property
    def subprocess_operations(self):

        global JOB_CALL_SCANNER_REQUEST_ON, JOB_CALL_SCANNER_REQUEST_OFF
        return {
            JOB_CALL_SCANNER_REQUEST_ON: self.request_on,
            JOB_CALL_SCANNER_REQUEST_OFF: self.request_off
        }
