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

import os
from hashlib import  md5
import random
from cPickle import loads, dumps, UnpickleableError
from ConfigParser import ConfigParser
import re

#
# INTERNAL DEPENDENCIES
#

import power_manager
import paths
import config_file
import logger
from scanomatic.generics.singleton import Singleton
import scanomatic.models.scanning_model as scanning_model

#
# CLASSES
#


class Config(Singleton):

    SCANNER_PATTERN = "Scanner {0}"
    POWER_DEFAULT = power_manager.POWER_MODES.Toggle

    def __init__(self, path=None):

        if path is None:
            self._paths = paths.Paths()
        else:
            self._paths = path

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
                    number_of_scans=9999,
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
        # TMP SOLUTION TO BIGGER PROBLEMS

        # VERSION HANDLING
        self.version_first_pass_change_1 = 0.997
        self.version_fixture_grid_history_change_1 = 0.998
        self.version_oldest_allow_fixture = 0.9991

        # SCANNER
        self.number_of_scanners = 3
        self.scanner_name_pattern = "Scanner {0}"
        self.scanner_names = list()
        self.scan_program = "scanimage"
        self.scan_program_version_flag = "-V"
        self._scanner_models = {
            self.SCANNER_PATTERN.format(i): 'EPSON V700' for i in range(1, 4)}

        # POWER MANAGER
        self._scanner_sockets = {
            self.SCANNER_PATTERN.format(i): i for i in range(1, 4)}

        self.pm_type = power_manager.POWER_MANAGER_TYPE.notInstalled
        self._pm_host = "192.168.0.100"
        self._pm_pwd = None
        self._pm_verify_name = False
        self._pm_MAC = None
        self._pm_name = "Server 1"

        # RPC SERVER
        self._rpc_port = None
        self._rpc_host = None
        self._config_rpc_admin = None
        self._server_config = None

        # HARDWARE RESOURCES
        self.resources_min_checks = 3
        self.resources_mem_min = 30
        self.resources_cpu_tot_free = 30
        self.resources_cpu_single = 75
        self.resources_cpu_n = 1

        # LOAD CONFIG FROM FILE
        self._load_config_from_file()

        self._set_pm_extras()

    def _load_config_from_file(self):

        self._config_file = config_file.ConfigFile(
            self._paths.config_main_app)

        scanners = self._config_file['number-of-scanners']
        if scanners is not None:
            self.number_of_scanners = scanners

        pm = self._config_file['pm-type']
        if pm is not None:
            try:
                self.pm_type = loads(pm)
            except UnpickleableError:
                self._logger.warning("Power Manager Type {0} not valid".format(
                    pm))

        experiments_root = self._config_file['experiments-root']
        if experiments_root is not None:
            self._paths.experiment_root = experiments_root

        pm_name = self._config_file['pm-name']
        if pm_name is not None:
            self._pm_name = pm_name

        pm_host = self._config_file['pm-host']
        if pm_host is not None:
            self._pm_host = pm_host

        pm_pwd = self._config_file['pm-pwd']
        if pm_pwd is not None:
            self._pm_pwd = pm_pwd

        pm_mac = self._config_file['pm-MAC']
        if pm_mac is not None:
            self._pm_MAC = pm_mac

    def _set_pm_extras(self):

        if self.pm_type == power_manager.POWER_MANAGER_TYPE.linuxUSB:

            self._PM = power_manager.USB_PM_LINUX
            self._pm_arguments = {
                'power_mode': self.POWER_DEFAULT
            }

        elif self.pm_type == power_manager.POWER_MANAGER_TYPE.LAN:
            self._PM = power_manager.LAN_PM

            self._pm_arguments = {
                'power_mode': self.POWER_DEFAULT,
                'host': self._pm_host,
                'password': self._pm_pwd,
                'verify_name': self._pm_verify_name,
                'pm_name': self._pm_name,
                'MAC': self._pm_MAC,
            }
        else:
            self._PM = power_manager.NO_PM
            self._pm_arguments = {
                'power_mode': self.POWER_DEFAULT
            }

    @staticmethod
    def _safe_config_get(cfg, section, item, default_value=None, vtype=None):

        try:

            default_value = cfg.get(section, item)
            if vtype is not None:
                default_value = vtype(default_value)

        except:

            pass

        return default_value

    @property
    def server_config(self):

        if self._server_config is None:
            self._server_config = ConfigParser(allow_no_value=True)
            self._server_config.readfp(open(self._paths.config_rpc))
            self._logger.info("Loaded RPC config from '{0}'".format(
                self._paths.config_rpc))
        return self._server_config

    @property
    def rpc_host(self):

        if self._rpc_host is None:
            host = self._safe_config_get(self.server_config, 'Communication', 'host', '127.0.0.1')
            self._rpc_host = host

        return self._rpc_host

    @property
    def rpc_port(self):

        if self._rpc_port is None:

            port = self._safe_config_get(self.server_config, 'Communication', 'port', 14547, int)
            self._rpc_port = port

        return self._rpc_port

    @property
    def rpc_admin(self):

        if self._config_rpc_admin is not None:
            return self._config_rpc_admin

        path = self._paths.config_rpc_admin

        if os.path.isfile(path):
            fh = open(path, 'r')
            admin = fh.read().strip()
            fh.close()
        else:
            admin = md5(str(random.random())).hexdigest()
            fh = open(path, 'w')
            fh.write(admin)
            fh.close()

        self._config_rpc_admin = admin

        return admin

    def set(self, key, value):

        if key == 'pm-type':
            if value in power_manager.hasValue(
                    power_manager.POWER_MANAGER_TYPE, value): 
                self.pm_type = power_manager.getEnumTypeFromValue(
                    power_manager.POWER_MANAGER_TYPE, value)
                self._set_pm_extras()

        elif key == 'number-of-scanners':

            if isinstance(value, int) and 0 <= value <= 4:
                self.number_of_scanners = value

    def save_settings(self):

        self._config_file['pm-type'] = dumps(self.pm_type)
        self._config_file['number-of-scanners'] = self.number_of_scanners
        self._config_file['experiments-root'] = self._paths.experiment_root
        self._config_file.save()

    def get_scanner_model(self, scanner):

        return self._scanner_models[self.get_scanner_name(scanner)]

    def get_scanner_name(self, scanner):

        if isinstance(scanner, int) and 0 < scanner < self.number_of_scanners:
            scanner = self.SCANNER_PATTERN.format(scanner)
        elif isinstance(scanner, str):
            numbers = map(int, re.findall(r'\d+', 'Scanner 1000'))
            if not(len(numbers) == 1 and 
                   0 < numbers[0] < self.number_of_scanners):
                
                return None
        else:
            return None

        return scanner
    
    def get_scanner_socket(self, scanner):

        scanner_name = self.get_scanner_name(scanner)
        if scanner_name in self._scanner_sockets:
            return self._scanner_sockets[scanner_name]
        else:
            self._logger.error("{0} not a scanner, can't return its power socket".format(scanner_name))
            return None

    def get_pm(self, scanner_name, **pm_kwargs):

        scanner_pm_socket = self.get_scanner_socket(scanner_name)
        if scanner_pm_socket is None:
            return power_manager.NO_PM("None")
        if len(pm_kwargs) == 0:
            pm_kwargs = self._pm_arguments

        self._logger.info(
            "Creating scanner PM for socket {0} and settings {1}".format(
                scanner_pm_socket, pm_kwargs))

        return self._PM(scanner_pm_socket, **pm_kwargs)

    def get_default_experiment_query(self):

        experiment_query = {
            '-f': None,  # FIXTURE: path to conf-file
            '-s': "",  # SCANNER to be used
            '-i': 20,  # INTERVAL in minutes
            '-n': 217,  # NUMBER OF SCANS
            '-m': "",  # PINNING LIST STRING
            '-r': self._paths.experiment_root,  # ROOT of experiments
            '-p': "",  # PREFIX for experiment
            '-d': "",  # DESCRIPTION
            '-c': "",  # PROJECT ID CODE
            '-u': "",  # UUID
            '--debug': 'info'  # LEVEL OF VERBOSITY
        }

        return experiment_query

    def get_default_analysis_query(self):

        analysis_query = {
            "-i": "",  # No default input file
            "-o": self._paths.experiment_analysis_relative_path,  # Default subdir
            # "-t" : 100,  # Time to set grid
            '--xml-short': 'True',  # Short output format
            '--xml-omit-compartments': 'background,cell',  # Only look at blob
            '--xml-omit-measures':
            'mean,median,IQR,IQR_mean,centroid,perimeter,area',  # only get pixelsum
            '--debug': 'info'  # Report everything that is info and above in seriousness
        }

        return analysis_query

    def get_min_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['min'])  

    def get_max_model(self, model, factory):

        return factory.create(**self._minMaxModels[type(model)]['max'])
