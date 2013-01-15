#!/usr/bin/env python
"""Resource Application Config"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

#
# INTERNAL DEPENDENCIES
#

import src.resource_power_manager as resource_power_manager
import src.resource_path as resource_path

#
# CLASSES
#

class Config(object):

    def __init__(self, paths=None):

        if paths is None:
            paths = resource_path.Paths()

        self._paths = paths

        #TMP SOLUTION TO BIGGER PROBLEMS

        #VERSION HANDLING
        self.version_first_pass_change_1 = 0.997

        #SCANNER
        self.number_of_scanners = 3
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

        self.pm_type = 'USB'
        self._pm_host = "192.168.0.100"
        self._pm_pwd = None
        self._pm_verify_name = False
        self._pm_MAC = None
        self._pm_name = "Server 1"

        if self.pm_type == 'USB':

            self._PM = resource_power_manager.USB_PM_LINUX
            self._pm_arguments={}

        elif self.pm_type == 'LAN':
            self._PM = resource_power_manager.LAN_PM

            self._pm_arguments = {'host': self._pm_host,
                'password': self._pm_pwd,
                'verify_name': self._pm_verify_name,
                'pm_name':self._pm_name,
                'MAC':self._pm_MAC,
                }

    def get_scanner_model(self, scanner):

        return self._scanner_models[scanner]

    def get_scanner_socket(self, scanner):

        return self._scanner_sockets[scanner]

    def get_pm(self, scanner_name, logger=None, **pm_kwargs):

        scanner_pm_socket = self._scanner_sockets[scanner_name]
        if pm_kwargs == {}:
            pm_kwargs = self._pm_arguments

        print "Creating scanner for socket {0} and settings {1}".format(scanner_pm_socket, pm_kwargs)
        return self._PM(scanner_pm_socket, logger=logger, **pm_kwargs)

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

        analysis_query =  {
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

