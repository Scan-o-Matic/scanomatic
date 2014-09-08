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

import ConfigParser
from subprocess import Popen, PIPE

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
import scanomatic.io.power_manager as power_manager


#TODO: Sets power to true when no pm

class Scanner_Manager(object):

    POWER_FLICKER_DELAY = 0.25

    def __init__(self):

        self._logger = logger.Logger("Scanner Manager")
        self._conf = app_config.Config()
        self._paths = paths.Paths()

        self._scannerStatus = ConfigParser.ConfigParser(
            allow_no_value=True)
        try:
            self._scannerStatus.readfp(open(
                self._paths.rpc_scanner_status, 'r'))
        except IOError:
            self._logger.info(
                "No scanner statuses previously known, starting fresh")

        self._scannerConfs = ConfigParser.ConfigParser(
            allow_no_value=True)
        try:
            self._scannerConfs.readfp(open(
                self._paths.config_scanners, 'r'))
        except IOError:
            self._logger.info(
                "No specific scanner configurations, all asumed default")

        self._set_powerManagers()

    def __contains__(self, scanner):

        try:
            self._conf.get_scanner_socket(scanner)
        except KeyError:
            return False

        return True

    def _get(self, scanner, key, defaultVal):
        
        scanner = self._conf.get_scanner_name(scanner)

        if not self._scannerStatus.has_section(scanner):
            self._scannerStatus.add_section(scanner)

        if self._scannerStatus.has_option(scanner, key):
            val = self._scannerStatus.get(scanner, key)
            if val == '':
                return None
            else:
                return val
        else:
            return defaultVal

    def _set(self, scanner, key, value):

        scanner = self._conf.get_scanner_name(scanner)

        if not self._scannerStatus.has_section(scanner):
            self._scannerStatus.add_section(scanner)

        if value is None:
            self._scannerStatus.set(scanner, key, '')
        elif isinstance(value, bool):
            self._scannerStatus.set(scanner, key, int(value))
        else:
            self._scannerStatus.set(scanner, key, str(value))

    def _set_powerManagers(self):

        self._pm = dict()
        for i in range(1, self._conf.number_of_scanners + 1):
            self._pm[i] = self._conf.get_pm(i)


    def _get_alive_scanners(self):

        p = Popen(["scanimage", "-L", "|",
            "sed", "-n", "-E", r"s/^.*device[^\`]*.(.*libusb[^\`']*).*$/\1/p",
            ], shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        
        if 'no SANE devices' in stderr:
            return []
        else:
            #TODO: why map str?
            return [s for s in map(str, stdout.split('\n')) if len(s) > 0]

    def _get_recorded_statuses(self):

        claim = dict()

        for sect in self._scannerStatus.sections():

            try:
                i = int(sect[-1])
            except (ValueError, TypeError):
                self._logger.error(
                    "Scanner Status File has corrupt section '{0}'".format(
                        sect))
                continue

            claim[i] = dict(
                    usb=self._get(sect, 'usb', ''),
                    power=bool(self._get(sect, 'power', False)))

        for i in range(1, self._conf.number_of_scanners + 1):
            if i not in claim:
                claim[i] = dict(usb=None, power=False)

        return claim

    def _updateStatus(self, claim):

        for c in claim:

            self._set(c, 'usb', claim[c]['usb'])
            self._set(c, 'power', claim[c]['power'])
        self._save()

    def _save(self):

        self._scannerStatus.write(open(self._paths.rpc_scanner_status, 'w'))

    def _rescue(self, usbList, claim):

        power = list(self.powerStatus)
        offs = 0

        for i in range(1, self._conf.number_of_scanners + 1):

            if power[i] and claim[i]['usb'] in usbList or not claim[i]['usb']:
                self.requestOff(i, updateClaim=False)
                power[i] = False
                offs += 1
                claim[i] = dict(usb=None, power=False)

        self._updateStatus(claim)

    def _match_scanners(self, scannerList):

        claim = self._get_recorded_statuses()
        noFounds = []
        while scannerList:
            usb = scannerList.pop()
            found = False
            for c in claim:
                if claim[c]['usb'] and claim[c]['usb'] == usb:
                    claim[c]['matched'] = True
                    found = True
                    break

            if not found:

                noFounds.push(usb)

        if len(noFounds) > 1:
            self._critical("More than one unclaimed scanner")
            self._rescue(noFounds, claim)
            return False

        for usb in noFounds:

            found = False
            for c in claim:

                if claim[c]['power'] and not claim[c]:

                    if not found:
                        claim[c]['usb'] = usb
                        found = True
                    else:
                        self._logger.critical(
                            "More than one scanner claiming to" +
                            "be on without matched usb")
                
                        self._rescue(noFounds, claim)
                        return False
                    
        self._updateStatus(claim)

        return sum(1 for c in claim 
                   if 'matched' in claim[c] and claim[c]['matched']) == \
               sum(1 for c in claim
                   if 'usb' in claim[c] and claim[c]['usb'])

    def _get_power_type(self, scanner):

        sName = self._conf.get_scanner_name(scanner)
        if self._scannerConfs.has_section(sName):

            powerType = self._scannerConfs.get(sName, 'powerType', 'SIMPLE')

        else:

            powerType = 'SIMPLE'

        return powerType
        
    def isOwner(self, scanner, jobID):

        owner = self._get(scanner, 'jobID')
        return owner is not None and owner == jobID

    def requestOn(self, scanner, jobID):

        if self.sync():

            if self._get(sName, 'power', False):
                usb = self._get(sName, 'usb', None)
                if usb:
                    return True

            if not self.isOwner(scanner, jobID):
                self._logger.error("Can't turn on, owner missmatch")
                return False

            powerType = self._get_power_type(scanner)

            self._set(sName, 'usb', None)

            if powerType == 'SIMPLE':
                self._pm[scanner].on()
            else:
                self._pm[scanner].on()
                time.sleep(self.POWER_FLICKER_DELAY)
                self._pm[scanner].off()

            self._set(sName, 'power', True)

            self._save()

        else:

            return False

    def requestOff(self, scanner, jobID, updateClaim=True):

        if not self.isOwner(scanner, jobID):
            self._logger.error("Can't turn on, owner missmatch")
            return False

        powerType = self._get_power_type(scanner)

        if powerType == 'SIMPLE':
            self._pm[scanner].off()
        else:
            self._pm[scanner].on()
            time.sleep(self.POWER_FLICKER_DELAY)
            self._pm[scanner].off()

        if updateClaim:
            self._set(scanner, 'usb', None)
            self._set(scanner, 'power', False)

            self._save()

        return True

    def requestClaim(self, scanner, pid, jobID):

        if scanner > self._conf.number_of_scanners:
            return False
        
        sName = self._conf.get_scanner_name(scanner)

        try:
            ownerProc = int(self._get(scanner, 'pid', -1))
        except (ValueError, TypeError):
            ownerProc = -1

        if ownerProc > 0 and pid != ownerProc:

            if psutil.pid_exists(ownerProc):

                self._logger.warning("Trying to claim {0} when claimed".format(
                    sName))
                return False

            else:
                self._logger.info(
                    "Releasing {0} since owner process is dead".format(
                        sName))

                self.releaseScanner(scanner)

        if self._get(scanner, "jobID", None):
            self._logger.warning("Overwriting previous jobID for {0}".format(
                sName))

        self._set(scanner, "pid", pid)
        self._set(scanner, "jobID", jobID)
        self._save()
        return True

    def releaseScanner(self, scanner):

        usb = self._get(scanner, "usb", None)
        if usb:
            self.requestOff(scanner)
        self._set(scanner, "pid", None)
        self._set(scanner, "jobID", None)
        self._save()
        return True

    def owner(self, scanner):

        return (self._get(scanner, "pid", None), 
                self._get(scanner, "jobID", None))
        
    def isOwner(self, scanner, jobID):

        return jobID is not None and self._get(scanner, "jobID", None) == jobID

    def usb(self, scanner, jobID):
        
        if jobID is None or jobID != self._get(scanner, "jobID", None):
            self._logger.warning("Incorrect jobID for scanner {0}".format(
                scanner))
            return False

        elif not self._get(scanner, "power", False):

            return None

        return self._get(scanner, "usb", None)

    def sync(self):

        return self._match_scanners(self._get_alive_scanners())

    @property
    def powerStatus(self):
        return (self._pm[i].status() for i in 
                range(1, self._conf.number_of_scanners + 1))
