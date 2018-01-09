import scanomatic.generics.model as model
import os
from uuid import uuid1


class VersionChangesModel(model.Model):

    def __init__(self, **kwargs):

        self.first_pass_change_1 = 0.997
        self.oldest_allow_fixture = 0.9991

        super(VersionChangesModel, self).__init__()


class RPCServerModel(model.Model):

    def __init__(self, port=None, host=None, admin=None, config=None):

        self.port = port
        self.host = host
        self.admin = admin
        self.config = config

        super(RPCServerModel, self).__init__()


class UIServerModel(model.Model):

    def __init__(self, port=5000, host="0.0.0.0", master_key=None):

        self.port = port
        self.host = host
        self.master_key = master_key if master_key else str(uuid1())
        super(UIServerModel, self).__init__()


class HardwareResourceLimitsModel(model.Model):

    def __init__(self, memory_minimum_percent=30, cpu_total_percent_free=30, cpu_single_free=75, cpu_free_count=1,
                 checks_pass_needed=3):

        self.memory_minimum_percent = memory_minimum_percent
        self.cpu_total_percent_free = cpu_total_percent_free
        self.cpu_single_free = cpu_single_free
        self.cpu_free_count = cpu_free_count
        self.checks_pass_needed = checks_pass_needed

        super(HardwareResourceLimitsModel, self).__init__()


class PathsModel(model.Model):

    def __init__(self, projects_root="/somprojects"):

        self.projects_root = projects_root

        super(PathsModel, self).__init__()


class MailModel(model.Model):

    def __init__(self, server=None, user=None, port=0, password=None, warn_scanning_done_minutes_before=30):

        self.server = server
        self.user = user
        self.port = port
        self.password = password
        self.warn_scanning_done_minutes_before = warn_scanning_done_minutes_before

        super(MailModel, self).__init__()


class ApplicationSettingsModel(model.Model):

    def __init__(self,
                 scanner_name_pattern="Scanner {0}",
                 rpc_server=None,
                 ui_server=None,
                 hardware_resource_limits=None,
                 computer_human_name="Unnamed Computer",
                 mail=None,
                 paths=None):

        self.versions = VersionChangesModel()
        self.rpc_server = rpc_server
        self.ui_server = ui_server
        self.hardware_resource_limits = hardware_resource_limits
        self.paths = paths
        self.computer_human_name = computer_human_name
        self.mail = mail

        super(ApplicationSettingsModel, self).__init__()
