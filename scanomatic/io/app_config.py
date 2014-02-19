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

#
# INTERNAL DEPENDENCIES
#

import power_manager
import paths
import config_file
import logger

#
# CLASSES
#


class Config(object):

    def __init__(self, path=None):

        if path is None:
            self._paths = paths.Paths()
        else:
            self._paths = path

        self._logger = logger.Logger("Application Config")

        #TMP SOLUTION TO BIGGER PROBLEMS

        #VERSION HANDLING
        self.version_first_pass_change_1 = 0.997
        self.version_fixture_grid_history_change_1 = 0.998
        self.version_oldest_allow_fixture = 0.9991

        #SCANNER
        self.number_of_scanners = 3
        self.scanner_name_pattern = "Scanner {0}"
        self.scanner_names = list()
        self.scan_program = "scanimage"
        self.scan_program_version_flag = "-V"
        self._scanner_models = {
            'Scanner 1': 'EPSON V700',
            'Scanner 2': 'EPSON V700',
            'Scanner 3': 'EPSON V700',
            'Scanner 4': 'EPSON V700'}

        #POWER MANAGER
        self._scanner_sockets = {
            'Scanner 1': 1,
            'Scanner 2': 2,
            'Scanner 3': 3,
            'Scanner 4': 4}

        self.pm_type = 'usb'
        self._pm_host = "192.168.0.100"
        self._pm_pwd = None
        self._pm_verify_name = False
        self._pm_MAC = None
        self._pm_name = "Server 1"

        #RPC SERVER
        self.rpc_port = 1420

        #HARDWARE RESOURCES
        self.resources_min_checks = 3
        self.resources_mem_min = 30
        self.resources_cpu_tot_free = 30
        self.resources_cpu_single = 75
        self.resources_cpu_n = 1

        #LOAD CONFIG FROM FILE
        self._load_config_from_file()

        self._set_pm_extras()

    def _load_config_from_file(self):

        self._config_file = config_file.Config_File(
            self._paths.config_main_app)

        scanners = self._config_file['number-of-scanners']
        if scanners is not None:
            self.number_of_scanners = scanners

        pm = self._config_file['pm-type']
        if pm is not None:
            self.pm_type = pm

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

        if self.pm_type == 'usb':

            self._PM = power_manager.USB_PM_LINUX
            self._pm_arguments = {}

        elif self.pm_type == 'lan':
            self._PM = power_manager.LAN_PM

            self._pm_arguments = {
                'host': self._pm_host,
                'password': self._pm_pwd,
                'verify_name': self._pm_verify_name,
                'pm_name': self._pm_name,
                'MAC': self._pm_MAC,
            }
        else:
            self._PM = power_manager.NO_PM
            self._pm_arguments = {}

    def set(self, key, value):

        if key == 'pm-type':
            if value in ('usb', 'lan', 'no'):
                self.pm_type = value
                self._set_pm_extras()

        elif key == 'number-of-scanners':

            if isinstance(value, int) and 0 <= value <= 4:
                self.number_of_scanners = value

    def save_settings(self):

        self._config_file['pm-type'] = self.pm_type
        self._config_file['number-of-scanners'] = self.number_of_scanners
        self._config_file['experiments-root'] = self._paths.experiment_root
        self._config_file.save()

    def get_scanner_model(self, scanner):

        return self._scanner_models[scanner]

    def get_scanner_socket(self, scanner):

        return self._scanner_sockets[scanner]

    def get_pm(self, scanner_name, **pm_kwargs):

        scanner_pm_socket = self._scanner_sockets[scanner_name]
        if pm_kwargs == {}:
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
            #"-t" : 100,  # Time to set grid
            '--xml-short': 'True',  # Short output format
            '--xml-omit-compartments': 'background,cell',  # Only look at blob
            '--xml-omit-measures':
            'mean,median,IQR,IQR_mean,centroid,perimeter,area',  # only get pixelsum
            '--debug': 'info'  # Report everything that is info and above in seriousness
        }

        return analysis_query
