__author__ = 'martin'

import scanomatic.models.settings_models as settings_models
from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.io.power_manager import POWER_MODES, POWER_MANAGER_TYPE


class VersionChangeFactory(AbstractModelFactory):

    MODEL = settings_models.VersionChangesModel
    STORE_SECTION_SERIALIZERS = {}

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_models.VersionChangesModel
        """
        return super(VersionChangeFactory, cls).create()


class PowerManagerFactory(AbstractModelFactory):

    MODEL = settings_models.PowerManagerModel
    STORE_SECTION_HEAD = "Power Manager"
    STORE_SECTION_SERIALIZERS = {
        "type": POWER_MANAGER_TYPE,
        "number_of_sockets": int,
        "host": str,
        "password": str,
        "name": str,
        "verify_name": bool,
        "mac": str,
        "power_mode": POWER_MODES
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_models.PowerManagerModel
        """

        return super(PowerManagerFactory, cls).create(**settings)


class RPCServerFactory(AbstractModelFactory):

    MODEL = settings_models.RPCServerModel
    STORE_SECTION_HEAD = "RPC Server (Main SoM Server)"
    STORE_SECTION_SERIALIZERS = {
        "port": int,
        "host": str,
        "admin": bool,
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_models.RPCServerModel
        """
        return super(RPCServerFactory, cls).create(**settings)


class UIServerFactory(AbstractModelFactory):

    MODEL = settings_models.UIServerModel
    STORE_SECTION_HEAD = "UI Server"
    STORE_SECTION_SERIALIZERS = {
        "port": int,
        "host": str,
        "local": bool
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_models.UIServerModel
        """

        return super(UIServerFactory, cls).create(**settings)


class HardwareResourceLimitsFactory(AbstractModelFactory):

    MODEL = settings_models.HardwareResourceLimitsModel
    STORE_SECTION_HEAD = "Hardware Resource Limits"
    STORE_SECTION_SERIALIZERS = {
        "memory_minimum_percent": float,
        "cpu_total_percent_free": float,
        "cpu_single_free": float,
        "cpu_free_count": int,
        "checks_pass_needed": int
    }

    @classmethod
    def create(cls, **settings):
        """
                :rtype : scanomatic.models.settings_models.HardwareResourceLimitsModel
        """
        return super(HardwareResourceLimitsFactory, cls).create(**settings)


class PathsFactory(AbstractModelFactory):

    MODEL = settings_models.PathsModel
    STORE_SECTION_HEAD = "Paths"
    STORE_SECTION_SERIALIZERS = {
        "projects_root": str,
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_models.PathsModel
        """
        return super(PathsFactory, cls).create(**settings)


class SMPTFactory(AbstractModelFactory):

    MODEL = settings_models.SMTPModel
    STORE_SECTION_HEAD = "SMTP"
    STORE_SECTION_SERIALIZERS = {
        "host": str,
        "port": int,
        "user": str,
        "password": lambda enforce=None, serialize=None, unserialize=None:
        str(serialize).encode('rot13') if serialize is not None else
        (str(unserialize.decode('rot13')) if unserialize is not None else
         (str(enforce) if enforce else "")),
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_model.SMPTModel
        """

        return super(SMPTFactory, cls).create(**settings)


class PipelineFactory(AbstractModelFactory):

    MODEL = settings_models.PipelineModel
    STORE_SECTION_HEAD = "Pipeline"
    STORE_SECTION_SERIALIZERS = {
        "mail_scanning_done_minutes_before": float
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.settings_model.PipelineModel
        """
        return super(PipelineFactory, cls).create(**settings)


class ApplicationSettingsFactory(AbstractModelFactory):

    MODEL = settings_models.ApplicationSettingsModel
    STORE_SECTION_HEAD = "General settings"

    _SUB_FACTORIES = {
        settings_models.PathsModel: PathsFactory,
        settings_models.HardwareResourceLimitsModel: HardwareResourceLimitsFactory,
        settings_models.PowerManagerModel: PowerManagerFactory,
        settings_models.RPCServerModel: RPCServerFactory,
        settings_models.UIServerModel: UIServerFactory,
        settings_models.SMTPModel: SMPTFactory,
        settings_models.PipelineModel: PipelineFactory
    }

    STORE_SECTION_SERIALIZERS = {
        "power_manager": settings_models.PowerManagerModel,
        "rpc_server": settings_models.RPCServerModel,
        "ui_server": settings_models.UIServerModel,
        "hardware_resource_limits": settings_models.HardwareResourceLimitsModel,
        "paths": settings_models.PathsModel,
        "smtp_model": settings_models.SMTPModel,
        "pipeline": settings_models.PipelineModel,
        "number_of_scanners": int,
        "scanner_name_pattern": str,
        "scan_program": str,
        "scan_program_version_flag": str,
        "scanner_models":
            lambda enforce=None, serialize=None, unserialize=None:
            ([serialize[name] for name in sorted(serialize.keys())] if isinstance(serialize, dict) else None)
            if serialize is not None else
            ((enforce if not isinstance(enforce, tuple) else list(enforce)) if enforce is not None else
             (unserialize if not isinstance(unserialize, tuple) else list(unserialize))),
    }

    @classmethod
    def create(cls, **settings):

        """
         :rtype : scanomatic.models.settings_models.ApplicationSettingsModel
        """

        cls.populate_with_default_submodels(settings)

        return super(ApplicationSettingsFactory, cls).create(**settings)