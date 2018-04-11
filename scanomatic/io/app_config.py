from __future__ import absolute_import

import os

#
# INTERNAL DEPENDENCIES
#

from . import logger, paths
from scanomatic.generics.singleton import SingeltonOneInit
from scanomatic.models.factories.settings_factories import (
    ApplicationSettingsFactory
)
import scanomatic.models.scanning_model as scanning_model


#
# CLASSES
#


class Config(SingeltonOneInit):

    SCANNER_PATTERN = "Scanner {0}"

    def __one_init__(self):

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

    @property
    def versions(self):
        """

        Returns: scanomatic.models.settings_models.VersionChangesModel

        """
        return self._settings.versions

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

    def model_copy(self):

        return ApplicationSettingsFactory.copy(self._settings)

    def get_scanner_name(self, scanner):

        if isinstance(scanner, int):
            scanner = self.SCANNER_PATTERN.format(scanner)

        return scanner

    def reload_settings(self):

        self._settings = ApplicationSettingsFactory.create()

        self._settings.rpc_server.host = os.environ.get(
            "SOM_BACKEND_HOST", '0.0.0.0')
        self._settings.rpc_server.port = int(os.environ.get(
            "SOM_BACKEND_PORT", 12451
        ))
        self._settings.rpc_server.admin = 'admin'

    def get_min_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['min'])

    def get_max_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['max'])
