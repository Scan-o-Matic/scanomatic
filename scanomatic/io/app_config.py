from ConfigParser import ConfigParser, NoOptionError, NoSectionError
import uuid
import os

#
# INTERNAL DEPENDENCIES
#

import power_manager
import paths
import logger
from scanomatic.generics.singleton import SingeltonOneInit
import scanomatic.models.scanning_model as scanning_model
from scanomatic.models.factories.settings_factories import (
    ApplicationSettingsFactory)

#
# CLASSES
#


class Config(SingeltonOneInit):

    SCANNER_PATTERN = "Scanner {0}"
    POWER_DEFAULT = power_manager.POWER_MODES.Toggle

    def __one_init__(self):

        self._paths = paths.Paths()

        self._logger = logger.Logger("Application Config")

        # TODO: Extend functionality to toggle to remote connect
        self._use_local_rpc_settings = True

        self._minMaxModels = {
            scanning_model.ScanningModel: {
                "min": dict(
                    time_between_scans=7.0,
                    number_of_scans=1,
                    project_name=None,
                    directory_containing_project=None,
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
                    description=None,
                    email=None,
                    pinning_formats=None,
                    fixture=None,
                    scanner=1),
                }

            }

        self.reload_settings()

    @staticmethod
    def _safe_get(conf_parser, section, key, default, type):

        try:
            return type(conf_parser.get(section, key))
        except (NoOptionError, NoSectionError):
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

    @computer_human_name.setter
    def computer_human_name(self, value):

        self._settings.computer_human_name = str(value)

    @property
    def number_of_scanners(self):
        """

        Returns: int

        """
        return self._settings.number_of_scanners

    @number_of_scanners.setter
    def number_of_scanners(self, value):

        if isinstance(value, int) and value >= 0:
            self._settings.number_of_scanners = value
            # TODO: Should update dependent values such as
            # length of scanner names
        else:
            self._logger.warning(
                "Refused to set number of scanners '{0}', only 0 or positive ints allowed".format(
                    value))

    @property
    def scanner_name_pattern(self):
        """

        Returns: str

        """
        return self._settings.scanner_name_pattern

    @scanner_name_pattern.setter
    def scanner_name_pattern(self, value):

        self._settings.scanner_name_pattern = str(value)

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

    def model_copy(self):

        return ApplicationSettingsFactory.copy(self._settings)

    def get_scanner_name(self, scanner):

        if isinstance(scanner, int) and 0 < scanner <= self.number_of_scanners:
            scanner = self.SCANNER_PATTERN.format(scanner)

        for s in self.scanner_names:
            if s == scanner:
                return scanner
        return None

    def reload_settings(self):

        if os.path.isfile(self._paths.config_main_app):
            try:
                self._settings = (
                    ApplicationSettingsFactory.serializer.load_first(
                        self._paths.config_main_app))
            except (IOError):
                self._settings = ApplicationSettingsFactory.create()
        else:
            self._settings = ApplicationSettingsFactory.create()

        if not self._settings:
            self._logger.info(
                "We'll use default settings for now.")
            self._settings = ApplicationSettingsFactory.create()

        if self._use_local_rpc_settings:
            self.apply_local_rpc_settings()

        self._PM = power_manager.get_pm_class(
            self._settings.power_manager.type)

    def apply_local_rpc_settings(self):

        rpc_conf = ConfigParser(allow_no_value=True)
        if not rpc_conf.read(self._paths.config_rpc):
            self._logger.warning(
                "Could not read from '{0}',".format(self._paths.config_rpc) +
                "though local settings were indicated to exist")

        self._settings.rpc_server.host = Config._safe_get(
            rpc_conf, "Communication", "host", '127.0.0.1', str)
        self._settings.rpc_server.port = Config._safe_get(
            rpc_conf, "Communication", "port", 12451, int)

        try:
            self._settings.rpc_server.admin = open(
                self._paths.config_rpc_admin, 'r').read().strip()
        except IOError:
            self._settings.rpc_server.admin = self._generate_admin_uuid()
        else:
            if not self._settings.rpc_server.admin:
                self._settings.rpc_server = self._generate_admin_uuid()

    def _generate_admin_uuid(self):

        val = str(uuid.uuid1())
        try:
            with open(self._paths.config_rpc_admin, 'w') as fh:
                fh.write(val)
                self._logger.info("New admin user identifier generated")
        except IOError:
            self._logger.critical(
                "Could not write to file '{0}'".format(
                    self._paths.config_rpc_admin) +
                ", you won't be able to perform any actions on Scan-o-Matic" +
                " until fixed." +
                " If you are really lucky, it works if rebooted, " +
                "but it seems your installation is corrupt.")
            return None

        return val

    def validate(self, bad_keys_out=None):
        """

        Args:
            bad_keys_out: list to hold keys with bad values
            :type bad_keys_out: list


        Returns:

        """
        if bad_keys_out is not None:
            try:
                while True:
                    bad_keys_out.pop()
            except IndexError:
                pass

        if not ApplicationSettingsFactory.validate(self._settings):
            self._logger.error(
                "There are invalid values in the current application settings,"
                "will not save and will reload last saved settings")

            if bad_keys_out is not None:

                for label in ApplicationSettingsFactory.get_invalid_names(
                        self._settings):

                    bad_keys_out.append(label)

            self.reload_settings()
            return False
        return True

    def save_current_settings(self):

        if self.validate():
            ApplicationSettingsFactory.serializer.purge_all(
                self._paths.config_main_app)

            ApplicationSettingsFactory.serializer.dump(
                self._settings, self._paths.config_main_app)

    def get_scanner_socket(self, scanner):

        scanner = self.get_scanner_name(scanner)

        if scanner:
            return self.scanner_sockes[scanner]
        return None

    def get_pm(self, socket):

        if socket is None or socket < 1 or socket > 4:
            self._logger.error("Socket '{0}' is unknown, {1} known".format(
                socket,
                "sockets 1-{0}".format(self.power_manager.number_of_sockets) if
                self.power_manager.number_of_sockets else "no sockets"))
            return power_manager.PowerManagerNull("None")

        self._logger.info(
            "Creating scanner PM for socked {0} and settings {1}".format(
                socket, dict(**self.power_manager)))

        return self._PM(socket, **self.power_manager)

    def get_min_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['min'])

    def get_max_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['max'])
