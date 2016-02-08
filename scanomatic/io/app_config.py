#!/usr/bin/env python
"""Resource Application Config"""
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


from ConfigParser import ConfigParser, NoOptionError

#
# INTERNAL DEPENDENCIES
#

import power_manager
import paths
import logger
from scanomatic.generics.singleton import SingeltonOneInit
import scanomatic.models.scanning_model as scanning_model
from scanomatic.models.factories.settings_factories import ApplicationSettingsFactory

#
# CLASSES
#


class Config(SingeltonOneInit):

    SCANNER_PATTERN = "Scanner {0}"
    POWER_DEFAULT = power_manager.POWER_MODES.Toggle

    def __one_init__(self):

        self._paths = paths.Paths()

        self._logger = logger.Logger("Application Config")

        self._minMaxModels = {
            scanning_model.ScanningModel: {
                "min": dict(
                    time_between_scans=7.0,
                    number_of_scans=1,
                    project_name=None,
                    directory_containing_project=None,
                    project_tag=None,
                    scanner_tag=None,
                    description=None,
                    email=None,
                    pinning_formats=None,
                    fixture=None,
                    scanner=1),
                "max": dict(
                    time_between_scans=None,
                    number_of_scans=999999,
                    project_name=None,
                    directory_containing_project=None,
                    project_tag=None,
                    scanner_tag=None,
                    description=None,
                    email=None,
                    pinning_formats=None,
                    fixture=None,
                    scanner=1),
                }

            }

        try:
            self._settings = ApplicationSettingsFactory.serializer.load(self._paths.config_main_app)[0]
        except (IndexError, IOError):
            self._settings = ApplicationSettingsFactory.create()

        rpc_conf = ConfigParser(allow_no_value=True)
        rpc_conf.read(self._paths.config_rpc)

        if not self._settings.rpc_server.host:
            self._settings.rpc_server.host = Config._safe_get(rpc_conf, "Communication", "host", '127.0.0.1', str)
        if not self._settings.rpc_server.port:
            self._settings.rpc_server.port = Config._safe_get(rpc_conf, "Communication", "port", 12451, int)
        if not self._settings.rpc_server.admin:
            try:
                self._settings.rpc_server.admin = open(self._paths.config_rpc_admin, 'r').read().strip()
            except IOError:
                pass

        self._PM = power_manager.get_pm_class(self._settings.power_manager.type)

    @staticmethod
    def _safe_get(conf_parser, section, key, default, type):

        try:
            return type(conf_parser.get(section, key))
        except NoOptionError:
            return default

    @property
    def versions(self):
        """

        Returns: scanomatic.models.settings_models.VersionChangesModel

        """
        return self._settings.versions

    @property
    def power_manager(self):
        """

        Returns: scanomatic.models.settings_models.PowerManagerModel

        """

        return self._settings.power_manager

    @property
    def rpc_server(self):
        """

        Returns: scanomatic.models.settings_models.RPCServerModel

        """
        return self._settings.rpc_server

    @property
    def ui_server(self):
        """

        Returns: scanomatic.models.settings_models.UIServerModel

        """
        return self._settings.ui_server

    @property
    def hardware_resource_limits(self):
        """

        Returns: scanomatic.models.settings_models.HardwareResourceLimitsModel

        """
        return self._settings.hardware_resource_limits

    @property
    def mail(self):
        """

        Returns: scanomatic.models.settings_models.MailModel

        """
        return self._settings.mail

    @property
    def paths(self):
        """

        Returns: scanomatic.models.settings_models.PathsModel

        """
        return self._settings.paths

    @property
    def computer_human_name(self):
        """

        Returns: str

        """
        return self._settings.computer_human_name

    @property
    def number_of_scanners(self):
        """

        Returns: int

        """
        return self._settings.number_of_scanners

    @property
    def scanner_name_pattern(self):
        """

        Returns: str

        """
        return self._settings.scanner_name_pattern

    @property
    def scanner_names(self):
        """

        Returns: [str]

        """
        return self._settings.scanner_names

    @property
    def scan_program(self):
        """

        Returns: str

        """
        return self._settings.scan_program

    @property
    def scan_program_version_flag(self):
        """

        Returns: str

        """
        return self._settings.scan_program_version_flag

    @property
    def scanner_models(self):
        """

        Returns: {str: str}

        """
        return self._settings.scanner_models

    @property
    def scanner_sockets(self):
        """

        Returns: {str: int}

        """
        return self._settings.scanner_sockets

    def get_scanner_name(self, scanner):

        if isinstance(scanner, int) and 0 < scanner <= self.number_of_scanners:
            scanner = self.SCANNER_PATTERN.format(scanner)

        for s in self.scanner_names:
            if s == scanner:
                return scanner
        return None
    
    def get_scanner_socket(self, scanner):

        scanner = self.get_scanner_name(scanner)

        if scanner:
            return self.scanner_sockes[scanner]
        return None

    def get_pm(self, socket):

        if socket is None:
            self._logger.error("Socket {0} is unknown".format(scanner))
            return power_manager.PowerManagerNull("None")


        self._logger.info(
            "Creating scanner PM for socked {0} and settings {1}".format(
                socket, dict(**self.power_manager)))

        return self._PM(socket, **self.power_manager)

    def get_min_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['min'])  

    def get_max_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['max'])
