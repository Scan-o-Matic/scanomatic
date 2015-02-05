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
from enum import EnumMeta
import re
from cPickle import loads, dumps
import psutil

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.app_config as app_config
import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
import scanomatic.io.fixtures as fixtures
import scanomatic.models.scanning_model as scanning_model
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory


class Scanner_Manager(object):


    def __init__(self):

        self._logger = logger.Logger("Scanner Manager")
        self._conf = app_config.Config()
        self._paths = paths.Paths()
        self._fixtures = fixtures.Fixtures()
        self._orphanUSBs = set()

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

    def __iter__(self):

        return iter(self._scannerStatus.sections())

    def _verifyDataStore(self, dataStore):

        if dataStore is None:
            dataStore = self._scannerStatus

        return dataStore

    def _verifySection(self, scanner, dataStore=None):

        dataStore = self._verifyDataStore(dataStore)

        if not dataStore.has_section(scanner):
            dataStore.add_section(scanner)

        return dataStore

    def _get(self, scanner, key, default=None, dataStore=None):
        
        dataStore = self._verifyDataStore(dataStore) 
        scanner = self._conf.get_scanner_name(scanner)
        self._verifySection(scanner, dataStore=dataStore)


        if dataStore.has_option(scanner, key):
            val = dataStore.get(scanner, key)
            if isinstance(val, EnumMeta):
                try:
                    val = loads(val)
                except:
                    self._logging.warning(
                        "Bad data for {0} on scanner {1} ({2})".format(
                            key, scanner, val))
            if val != '':
                return val

        return default 

    def _set(self, scanner, key, value, dataStore=None):

        dataStore = self._verifyDataStore(dataStore) 
        scanner = self._conf.get_scanner_name(scanner)
        self._verifySection(scanner, dataStore=dataStore)

        if value is None:
            dataStore.set(scanner, key, '')
        elif isinstance(value, bool):
            dataStore.set(scanner, key, str(int(value)))
        elif isinstance(value, EnumMeta):
            dataStore.set(scanner, key, dumps(value))
        else:
            dataStore.set(scanner, key, str(value))

    def _set_powerManagers(self):

        self._pm = dict()
        for i in range(1, self._conf.number_of_scanners + 1):
            self._pm[self._conf.get_scanner_name(i)] = self._conf.get_pm(i)


    def _get_alive_scanners(self):

        p = Popen(["scanimage", "-L"], shell=False, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        return re.findall(r"device[^\`]*.(.*libusb[^\`']*)", stdout)

    def _get_recorded_statuses(self):

        claims = dict()

        for scanner in self:

            claims[scanner] = dict(
                    usb=self.getUSB(scanner, default=None),
                    power=self.getPower(scanner))

        return claims

    def _updateStatus(self, claim):

        for c in claim:
            self._set(c, 'usb', claim[c]['usb'])
            self._set(c, 'power', claim[c]['power'])
        self._save()

    def _save(self):

        self._scannerStatus.write(open(self._paths.rpc_scanner_status, 'w'))

    def _rescue(self, orphanUSBs, claim):

        self._orphanUSBs = self._orphanUSBs.union(orphanUSBs)

        power = self.powerStatus
    
        for scanner in self:

            couldHaveOrClaimsPower = (power[scanner] or claim[scanner]['power'])
            hasNoneOrBadUSB =  (claim[scanner]['usb'] in orphanUSBs                                                               
                                         or not claim[scanner]['usb'])

            if (couldHaveOrClaimsPower and hasNoneOrBadUSB):

                if self.requestOff(scanner, self._get(scanner, 'jobID', ''),
                                   updateClaim=False):

                    claim[scanner] = dict(usb=None, power=False)

        self._updateStatus(claim)

    def _match_scanners(self, currentUSBs):

        claim = self._get_recorded_statuses()
        self._updateOrphanedUSBs(currentUSBs)
        currentUSBs = self._filterOutOrphanedUSBs(currentUSBs)
        unknownUSBs = self._getUnknownUSBs(currentUSBs, claim)

        if (self._canAssignUnknownUSBs(unknownUSBs)):
            if self._assignUnknownUSBtoClaim(unknownUSBs, claim):
                self._rescue(unknownUSBs, claim)
                return False
        else:
            self._rescue(unknownUSBs, claim)
            return False
        


        self._updateStatus(claim)

        return self._getUSBclaimIsValid(claim)


    def _getUSBclaimIsValid(self, claim):

        scannersClaimingToKnowUSBs = sum(1 for c in claim 
                   if 'matched' in claim[c] and claim[c]['matched'])
        scannersWithAssignedUSBs = sum(1 for c in claim
                   if 'usb' in claim[c] and claim[c]['usb'])

        return scannersWithAssignedUSBs == scannersClaimingToKnowUSBs

    def _canAssignUnknownUSBs(self, unknownUSBs):

        if len(unknownUSBs) > 1:
            self._logger.critical("More than one unclaimed scanner {0}".format(
                unknownUSBs))
            return False
        return True

    def _assignUnknownUSBtoClaim(self, unknownUSBs, claim):

        if len(unknownUSBs)==1:

            usb = unknownUSBs[0]

            if (not self._setUSBtoClaim(usb, claim)):

                return self._setUSBtoScannerThatCouldBeOn(usb, claim)

        return len(unknownUSBs) == 0

    def _setUSBtoScannerThatCouldBeOn(self, usb, claim):

        powers = self.powerStatus
        if (sum(powers.values()) == 1):

            scanner = tuple(scanner for scanner in powers if powers[scanner])[0]

            if (self._pm[scanner].sureToHavePower()):
                claim[scanner]['power'] = True
                claim[scanner]['usb'] = usb
            else:
                self._logger.critical(
                    "There's one scanner on {0}, but can't safely assign it to {1}".format(
                        usb, scanner))
                return False


        else:

            self._logger.critical(
                "There's one scanner on {0}, but non claims to be".format(
                    usb))
            return False

        return True

    def _setUSBtoClaim(self, usb, claim):

        foundScannerWithPowerAndNoUSB = None
        for c in claim:

            if claim[c]['power'] and not claim[c]['usb']:

                if not foundScannerWithPowerAndNoUSB:
                    claim[c]['usb'] = usb
                    foundScannerWithPowerAndNoUSB = c 
                else:
                    self._logger.critical(
                        "More than one scanner claiming to" +
                        "be on without matched usb")

                    claim[foundScannerWithPowerAndNoUSB]['usb'] = ''
                    return False

        return foundScannerWithPowerAndNoUSB != None
                    
    def _getUnknownUSBs(self, currentUSBs, claim):

        unknownUSBs= []
        while currentUSBs:
            usb = currentUSBs.pop()
            found = False
            for c in claim:
                if claim[c]['usb'] and claim[c]['usb'] == usb:
                    claim[c]['matched'] = True
                    found = True
                    break

            if not found:

                unknownUSBs.append(usb)

        return unknownUSBs

    def _updateOrphanedUSBs(self, currentUSBs):

        self._orphanUSBs = self._orphanUSBs.intersection(currentUSBs)

    def _filterOutOrphanedUSBs(self, usbs):

        return set(usbs).difference(self._orphanUSBs)

    def isOwner(self, scanner, jobID):

        return jobID == self._get(scanner, 'jobID', '')

    def requestOn(self, scanner, jobID):

        if self.sync():

            if self.getUSB(scanner, default=False):
                #Scanner is already On and connected
                return True

            if self.isOwner(scanner, jobID) is False:
                self._logger.error(
                    "Can't turn on {2}, owner missmatch ('{0}'!='{1}')".format(
                        jobID, self._get(scanner, "jobID", ""), scanner))
                return False

            self._set(scanner, 'usb', None)

            success = self._pm[scanner].powerUpScanner()

            self._set(scanner, 'power', success)

            self._save()
            return success

        else:

            return False

    def requestOff(self, scanner, jobID, updateClaim=True):

        if self.isOwner(scanner, jobID) is False:
            self._logger.error(
                "Can't turn off {2}, owner missmatch ('{0}'!='{1}')".format(
                    jobID, self._get(scanner, "jobID", ""), scanner))
            return False

        success = self._pm[scanner].powerDownScanner()

        if success and updateClaim:
            self._set(scanner, 'usb', None)
            self._set(scanner, 'power', False)

            self._save()

        return success 

    def requestClaim(self, rpcJobModel):

        content_model = rpcJobModel.contentModel
        scanner = self._conf.get_scanner_name(content_model.scanner)

        if scanner not in self:
            self._logger.warning("Unknown scanner referenced ({0})".format(
                scanner))
            return False

        try:
            ownerProc = int(self._get(scanner, 'pid', -1))
        except (ValueError, TypeError):
            ownerProc = -1

        if ownerProc > 0 and rpcJobModel.pid != ownerProc:

            if psutil.pid_exists(ownerProc):

                self._logger.warning("Trying to claim {0} when claimed".format(
                    scanner))
                return False

            else:
                self._logger.info(
                    "Releasing {0} since owner process is dead".format(
                        scanner))

                self.releaseScanner(rpcJobModel)

        if self._get(scanner, "jobID", None):
            self._logger.warning("Overwriting previous jobID for {0}".format(
                scanner))

        self._set(scanner, "pid", rpcJobModel.pid)
        self._set(scanner, "jobID", rpcJobModel.id)
        self._save()
        return True

    def releaseScanner(self, rpcJobModel):

        content_model = rpcJobModel.contentModel

        if self.getUSB(rpcJobModel, default=False):
            self.requestOff(rpcJobModel)
        self._set(content_model.scanner, "pid", None)
        self._set(content_model.scanner, "jobID", None)

        if (content_model.status == scanning_model.JOB_STATUS.Running or
                content_model.status == scanning_model.JOB_STATUS.Queued or
                content_model.status == scanning_model.JOB_STATUS.Requested):

            content_model.status = scanning_model.JOB_STATUS.Done

        self._save()
        return True

    def owner(self, scanner):

        return (int(self._get(scanner, "pid", None)),
                self._get(scanner, "jobID", None))
        
    def updatePid(self, rpcJobModel):

        scanner = rpcJobModel.contentModel.scanner
        scanner = self._conf.get_scanner_name(scanner)

        if RPC_Job_Model_Factory.validate(rpcJobModel):
            return False

        for scanner in self._scannerStatus.sections():
            job = self._get(scanner, "jobID", None)
            if job == rpcJobModel.id:
                self._set(scanner, "pid", rpcJobModel.pid)
                self._save()
                return True

        return False

    def getUSB(self, rpcJobModel, default=''):
        """Gets the usb that a scanner is connected on.
        """
        
        """
        if jobID is None or jobID != self._get(scanner, "jobID", None):
            self._logger.warning("Incorrect jobID for scanner {0}".format(
                scanner))
            return False
        """
        
        scanner = rpcJobModel.contentModel.scanner

        if not self._get(scanner, "power", False):

            return default

        return self._get(scanner, "usb", default)

    def getPower(self, scanner):

        return bool(int(self._get(scanner, "power", 0)))

    def sync(self):

        return self._match_scanners(self._get_alive_scanners())

    def fixtureExists(self, fixtureName):

        return fixtureName in self._fixtures

    def getFixtureNames(self):

        return self._fixtures.get_names()

    def getStatus(self, scanner):

        return dict(
            name=self._conf.get_scanner_name(scanner),
            pid=self._get(scanner, "pid", ""),
            power=self.getPower(scanner),
            owner=self._get(scanner, "jobID", ""),
            usb=self.getUSB(scanner))

    @property
    def powerStatus(self):
        return {scanner: self._pm[scanner].couldHavePower() 
                for scanner in self}
