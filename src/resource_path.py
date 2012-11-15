#!/usr/bin/env python
"""Resource Paths"""
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

import os

#
# CLASSES
#


class Paths(object):

    def __init__(self, program_path, config_file=None):

        self.root = program_path
        self.scanomatic = self.root + os.sep + "run_analysis.py"
        self.src = self.root + os.sep + "src"
        self.analysis = self.src + os.sep + "analysis.py"
        self.config = self.src + os.sep + "config"
        self.fixtures = self.config + os.sep + "fixtures"
        self.images = self.src + os.sep + "images"
        self.marker = self.images + os.sep + "orientation_marker_150dpi.png" 
        self.log = self.root + os.sep + "log"
        self.experiment_root = os.path.expanduser("~") + os.sep + "Documents"
        self.experiment_analysis_relative = "analysis"
        self.experiment_analysis_file_name = "analysis.log"
        self.experiment_first_pass_analysis_relative = "{0}.1_pass.analysis"
        self.experiment_first_pass_log_relative = ".1_pass.log"

