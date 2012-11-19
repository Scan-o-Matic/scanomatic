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

#
# CLASSES
#

class Config(object):

    def __init__(self, paths):

        self._paths = paths

        #TMP SOLUTION TO BIGGER PROBLEMS
        self.number_of_scanners = 3
        self.scanner_names = list()
        self.scan_program = "scanimage"
        self.scan_program_version_flag = "-V"

        self._scanner_names = {
            'Scanner 1': 1,
            'Scanner 2': 2,
            'Scanner 3': 3,
            'Scanner 4': 4}

        self.pm_type = 'LAN'
        self._pm_host = None
        self._pm_pwd = None
        self._pm_verify_name = False
        self._pm_MAC = None
        self._pm_name = "Server 1"

        if self.pm_type == 'USB':

            self._PM = resource_power_manager.USB_PM_LINX
            self._pm_arguments={}

        elif self.pm_type == 'LAN':
            self._PM = resource_power_manager.LAN_PM

            self._pm_arguments = {'host': self._pm_host,
                'password': self._pm_pwd,
                'verify_name': self._pm_verify_name,
                'pm_name':self._pm_name,
                'MAC':self._pm_MAC,
                'DMS':None}

    def get_pm(self, self._scanner_names[scanner_name],
            self._pm_arguments):


        return self._PM(scanner_name)

    def get_default_analysis_query(self):

        analysis_query =  {
            "-i": "",  # No default input file
            "-o": self._paths.experiment_analysis_relative_path,  # Default subdir
            "-t" : 100,  # Time to set grid
            '--xml-short': 'True',  # Short output format
            '--xml-omit-compartments': 'background,cell',  # Only look at blob
            '--xml-omit-measures':
            'mean,median,IQR,IQR_mean,centroid,perimeter,area',  # only get pixelsum
            '--debug': 'info'  # Report everything that is info and above in seriousness
            }

        return analysis_query

