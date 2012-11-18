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

