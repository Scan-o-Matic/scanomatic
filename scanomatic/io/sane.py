#!/usr/bin/env python
"""This module contains a class for obtaining images using SANE (Linux)."""

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
import os
import copy
import time
from itertools import chain
from enum import Enum
from types import StringTypes

#
# INTERNAL DEPENDENCIES
#

from logger import Logger

#
# Functions
#


def get_alive_scanners():

    p = Popen(["scanimage", SCAN_FLAGS.ListScanners.value], shell=False, stdout=PIPE, stderr=PIPE)
    stdout, _ = p.communicate()
    return re.findall(r'device[^\`]*.(.*libusb[^\`\']*)\' is a (.*) scanner', stdout)

#
# CLASSES
#


class SCAN_MODES(Enum):

    TPU = 0
    COLOR = 1


class SCANNER_DATA(Enum):

    SANEBackend = 0
    Aliases = 1
    DefaultTransparencyWord = 2
    WaitBeforeScan = 3


class SCAN_FLAGS(Enum):

    Source = "--source"
    Format = "--format"
    Resolution = "--resolution"
    Mode = "--mode"
    Left = "-l"
    Top = "-t"
    Width = "-x"
    Height = "-y"
    Depth = "--depth"
    Device = "-d"
    Help = '--help'
    ListScanners = '-L'


class SaneBase(object):

    _TRANSPARENCY_WORDS = {"TPU", "Transparency"}
    _SETTINGS_ORDER = (SCAN_FLAGS.Source, SCAN_FLAGS.Format, SCAN_FLAGS.Resolution, SCAN_FLAGS.Mode, SCAN_FLAGS.Left,
                       SCAN_FLAGS.Top, SCAN_FLAGS.Width, SCAN_FLAGS.Height, SCAN_FLAGS.Depth)

    _SETTINGS_REPOSITORY = {
        "EPSON V700": {
            SCANNER_DATA.WaitBeforeScan: 0,
            SCANNER_DATA.SANEBackend: 'epson2',
            SCANNER_DATA.Aliases: ('GT-X900', 'V700'),
            SCANNER_DATA.DefaultTransparencyWord: 'TPU8x10',
            SCAN_MODES.TPU: {
                SCAN_FLAGS.Source: "Transparency", SCAN_FLAGS.Format: "tiff",
                SCAN_FLAGS.Resolution: "600", SCAN_FLAGS.Mode: "Gray", SCAN_FLAGS.Left: "0",
                SCAN_FLAGS.Top: "0", SCAN_FLAGS.Width: "203.2", SCAN_FLAGS.Height: "254", SCAN_FLAGS.Depth: "8"},
            SCAN_MODES.COLOR: {
                SCAN_FLAGS.Source: "Flatbed", SCAN_FLAGS.Format: "tiff",
                SCAN_FLAGS.Resolution: "300", SCAN_FLAGS.Mode: "Color", SCAN_FLAGS.Left: "0",
                SCAN_FLAGS.Top: "0", SCAN_FLAGS.Width: "215.9", SCAN_FLAGS.Height: "297.18",
                SCAN_FLAGS.Depth: "8"}},
        "EPSON V800": {
            SCANNER_DATA.WaitBeforeScan: 3,
            SCANNER_DATA.SANEBackend: 'epson2',
            SCANNER_DATA.Aliases: ('GT-X980', 'V800'),
            SCANNER_DATA.DefaultTransparencyWord: 'TPU8x10',
            SCAN_MODES.TPU: {
                SCAN_FLAGS.Source: "Transparency", SCAN_FLAGS.Format: "tiff",
                SCAN_FLAGS.Resolution: "600", SCAN_FLAGS.Mode: "Gray", SCAN_FLAGS.Left: "0",
                SCAN_FLAGS.Top: "0", SCAN_FLAGS.Width: "203.2", SCAN_FLAGS.Height: "254", SCAN_FLAGS.Depth: "8"},
            SCAN_MODES.COLOR: {
                SCAN_FLAGS.Source: "Flatbed", SCAN_FLAGS.Format: "tiff",
                SCAN_FLAGS.Resolution: "300", SCAN_FLAGS.Mode: "Color", SCAN_FLAGS.Left: "0",
                SCAN_FLAGS.Top: "0", SCAN_FLAGS.Width: "215.9", SCAN_FLAGS.Height: "297.18",
                SCAN_FLAGS.Depth: "8"}}}

    _PROGRAM = "scanimage"
    _SOURCE_SEPARATOR = "|"
    _SOURCE_PATTERN = re.compile(r'--source ([^\n[]+)')

    def __init__(self, model, scan_mode):

        self._logger = Logger("SANE")

        self.next_file_name = None

        self._scan_settings = None

        self._model = SaneBase._get_model(model, self._logger)
        self._scan_mode = SaneBase._get_mode(scan_mode, self._model, self._logger)
        self._verified_settings = False
        self._name_for_transparency_mode = None
        self._scan_settings = self._get_copy_of_settings()

    @property
    def model(self):

        return self._model

    @model.setter
    def model(self, model_name):

        model = SaneBase._get_model(model_name, self._logger)

        if not model:
            self._logger.error("'{0}' is not a recognized scanner model".format(model_name))

        elif model != self._model:

            self._model = model
            self._name_for_transparency_mode = None
            self._scan_settings = self._get_copy_of_settings()
            self._logger.info("Updated scanner model to {0}".format(model))

    @classmethod
    def _get_model(cls, model, logger):

        model = model.upper()

        if model not in SaneBase._SETTINGS_REPOSITORY:

            for m, settings in SaneBase._SETTINGS_REPOSITORY.iteritems():

                if any(alias.upper() in model for alias in settings[SCANNER_DATA.Aliases]):
                    return m

            logger.critical("Model {0} unknown, only have settings for {1}".format(
                model, SaneBase._SETTINGS_REPOSITORY.keys()))

            return None

        return model

    @classmethod
    def _get_mode(cls, scan_mode, model, logger):

        if not model:
            return None

        if isinstance(scan_mode, StringTypes):
            scan_mode = scan_mode.upper()
            if scan_mode == "COLOUR":
                scan_mode = "COLOR"

            try:
                scan_mode = SCAN_MODES[scan_mode]
            except KeyError:
                pass

        if scan_mode not in SaneBase._SETTINGS_REPOSITORY[model]:
            logger.critical("Unknown scan-mode \"{0}\" for {1}, only knows {2}".format(
                scan_mode, model, SaneBase._SETTINGS_REPOSITORY[model].keys()))

        return scan_mode

    def _verify_mode_source(self, device):

        def _transparency_word(word):
            return any(True for tpu in SaneBase._TRANSPARENCY_WORDS if word.upper().startswith(tpu.upper()))

        def _select_preferred(words):

            default_word = SaneBase._SETTINGS_REPOSITORY[self._model][SCANNER_DATA.DefaultTransparencyWord]
            if self._scan_settings and default_word.upper() in (w.upper() for w in words):

                return tuple(w for w in words if w.upper() ==
                             default_word.upper())[0]

            return words[-1]

        proc = Popen([SaneBase._PROGRAM, SCAN_FLAGS.Device.value, device, SCAN_FLAGS.Help.value],
                     stdout=PIPE, stderr=PIPE, shell=False)
        stdout, _ = proc.communicate()
        try:
            sources = SaneBase._SOURCE_PATTERN.findall(stdout)
            self._logger.info("Sources matches are {0} for device {1}".format(sources, device))
            sources = sources[0].split(SaneBase._SOURCE_SEPARATOR)
            sources = tuple(source.strip() for source in sources)
            self._logger.info("Possible modes reported for scanner {0} are {1}".format(device, sources))
            tpu_sources = tuple(source for source in sources if _transparency_word(source))

            self._name_for_transparency_mode = _select_preferred(tpu_sources)
            return True

        except (TypeError, IndexError):
            self._logger.critical("Can't get information about the scanner {0} ({1}, {2})".format(
                device, self._model, self._scan_mode))
            return False

    def _update_mode_source(self):

        default_word = SaneBase._SETTINGS_REPOSITORY[self._model][SCANNER_DATA.DefaultTransparencyWord]
        if self._scan_mode is SCAN_MODES.TPU and not self._verified_settings:
            self._scan_settings[SCAN_FLAGS.Source] = self._name_for_transparency_mode if \
                self._name_for_transparency_mode else default_word

    def _get_copy_of_settings(self):

        if self._name_for_transparency_mode is None and self._scan_mode is not SCAN_MODES.COLOR:
            self._logger.info("Will verify correct transparency setting for first scan")

        try:
            return copy.deepcopy(SaneBase._SETTINGS_REPOSITORY[self._model][self._scan_mode])
        except KeyError:
            self._logger.critical("Without know settings, no scanning possible")
            return None

    def _get_scan_instructions(self, prepend=None):

        def _dict_to_tuple(d, key_order=None):

            if key_order is None:
                key_order = d.keys()
            else:
                key_order = tuple(key for key in key_order if key in d)
                key_order += tuple(set(d.keys()).difference(key_order))

            return tuple(chain(*((key.value, d[key]) for key in key_order)))

        program = (SaneBase._PROGRAM,)
        if prepend:
            prepend_settings = _dict_to_tuple(prepend)
        else:
            prepend_settings = tuple()

        settings = _dict_to_tuple(self._scan_settings, key_order=SaneBase._SETTINGS_ORDER)
        return program + prepend_settings + settings

    def OpenScanner(self, mainWindow=None, ProductName=None, UseCallback=False):
        pass

    def Terminate(self):
        pass

    def AcquireNatively(self, scanner=None, filename=None, **kwargs):
        return self.AcquireByFile(scanner=None, filename=filename, **kwargs)

    def _setup_settings(self, scanner, max_wait=10):

        start_time = time.time()

        while time.time() - start_time < max_wait:
            success = self._verify_mode_source(scanner)
            if success:
                self._update_mode_source()
                self._verified_settings = True
                return True
            elif self._scan_mode is SCAN_MODES.TPU:
                time.sleep(0.5)
            else:
                self._verified_settings = True
                return True

        return False

    def AcquireByFile(self, scanner=None, filename=None, **kwargs):

        if self._scan_settings is None:
            self._logger.critical("Without settings no scan possible.")
            return False

        elif self._verified_settings is False:
            self._setup_settings(scanner)

        if filename is not None:
            self.next_file_name = filename
        else:
            filename = self.next_file_name

        if filename:
            self._logger.info("Scanning {0}".format(filename))

            returncode = 0
            stderr = ""

            try:
                with open(filename, 'w') as im:
                    if scanner:
                        preprend_settings = {SCAN_FLAGS.Device: scanner}
                    else:
                        preprend_settings = None

                    scan_query = self._get_scan_instructions(prepend=preprend_settings)
                    self._logger.info("Scan-query is:\n{0}".format(" ".join(scan_query)))

                    if self._scan_settings[SCANNER_DATA.WaitBeforeScan]:
                        time.sleep(self._scan_settings[SCANNER_DATA.WaitBeforeScan])

                    scan_proc = Popen(scan_query, stdout=im, stderr=PIPE, shell=False)
                    _, stderr = scan_proc.communicate()

                    returncode = scan_proc.returncode

            except IOError:
                self._logger.error("Could not write to file: {0}".format(filename))
                returncode = -1

            else:

                if returncode:
                    self._logger.critical("scanimage produced return-code {0}".format(returncode))
                    self._logger.critical("Standard error from scanimage:\n\n{0}\n\n".format(stderr))
                    os.remove(filename)
                else:
                    self._logger.error("Error occurred while scanning but scanimage not reporting error code." +
                                       " stderr of scanimage shows:\n\n{0}\n\n".format(stderr))
                    returncode = -1

            return returncode == 0

        else:
            return False
