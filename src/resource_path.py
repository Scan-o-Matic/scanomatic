#!/usr/bin/env python
"""Resource Paths"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import os

#
# EXCEPTIONS
#

class Invalid_Root(Exception): pass

#
# CLASSES
#


class Paths(object):

    def __init__(self, program_path=None, root=None, src_path=None,
                config_file=None):

        if root is None:

            if program_path is not None:
                root = os.path.dirname(os.path.abspath(program_path))
            elif src_path is not None:
                root = os.path.dirname(os.sep.join(src_path.split(os.sep)[:-1]))
            else:
                root = os.sep.join(
                os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])


        self.root = root 
        if os.path.isdir(root) == False:
            raise Invalid_Root(root)
        self.src = self.root + os.sep + "src"
        if os.path.isdir(self.src) == False:
            raise Invalid_Root(root)
        self.config = self.root + os.sep + "config"
        if os.path.isdir(self.config) == False:
            raise Invalid_Root(root)
        self.fixtures = self.config + os.sep + "fixtures"
        self.images = self.src + os.sep + "images"

        #INSTALL
        self.desktop_file = "scan-o-matic.desktop"
        self.desktop_file_path = os.sep.join((self.config, "desktop_icon",
            self.desktop_file))

        #RUN-files
        self.scanomatic = self.root + os.sep + "run_scan_o_matic.py"
        self.analysis = os.sep.join((self.root, "run_analysis.py"))
        self.experiment = os.sep.join((self.root, "run_experiment.py"))

        #CONFIG
        self.config_main_app = os.sep.join((self.config, 'main.config'))
        self.config_mac = os.sep.join((self.config, 'mac_address.config'))
 
        #IMAGES
        self.marker = self.images + os.sep + "orientation_marker_150dpi.png" 
        self.martin = os.sep.join((self.images, "martin3.png"))
        self.logo = os.sep.join((self.images, "scan-o-matic.png"))

        #FIXTURE_FILES
        self.fixture_conf_file_suffix = ".config"
        self.fixture_conf_file_rel_pattern = "{0}" + \
            self.fixture_conf_file_suffix
        self.fixture_image_file_rel_pattern = "{0}.npy"
        self.fixture_conf_file_pattern = os.sep.join((
            self.fixtures, self.fixture_conf_file_rel_pattern))
        self.fixture_image_file_pattern = os.sep.join((
            self.fixtures, self.fixture_image_file_rel_pattern))
        self.fixture_tmp_scan_image = \
            self.fixture_image_file_pattern.format(".tmp")

        #LOG
        self.log = self.root + os.sep + "log"
        self.log_scanner_out = os.sep.join((self.log, "scanner_{0}.stdout"))
        self.log_scanner_err = os.sep.join((self.log, "scanner_{0}.stderr"))
        self.log_main_out = os.sep.join((self.log, "main.stdout"))
        self.log_main_err = os.sep.join((self.log, "main.stderr"))
        self._last_analysis_log_index = 0
        self.log_analysis_out = os.sep.join((self.log, "analysis_{0}.stdout"))
        self.log_analysis_err = os.sep.join((self.log, "analysis_{0}.stderr"))

        #EXPERIMENT
        self.experiment_root = os.path.expanduser("~") + os.sep + "Documents"
        self.experiment_scan_image_relative_pattern = "{0}_{1}.tiff"
        self.experiment_analysis_relative_path = "analysis"
        self.experiment_analysis_file_name = "analysis.log"

        #ANALSYS FILES
        self.analysis_polynomial = os.sep.join((self.config,
             "calibration.polynomials"))

        self.experiment_first_pass_analysis_relative = "{0}.1_pass.analysis"
        self.experiment_first_pass_log_relative = ".1_pass.log"
        self.experiment_local_fixturename = \
            self.fixture_conf_file_rel_pattern.format("fixture")
        self.experiment_grid_image_pattern = "grid___origin_plate_{0}.svg"

        #LOCK FILES
        self.lock_root = os.sep.join((os.path.expanduser("~"), ".scan_o_matic"))
        self.lock_power_up_new_scanner = self.lock_root + ".new_scanner.lock"
        self.lock_scanner_pattern = self.lock_root + ".scanner.{0}.lock"
        self.lock_scanner_addresses = self.lock_root + ".addresses.lock"

        #EXPERIMENT FILE PIPE
        self.experiment_stdin = self.lock_root + ".{0}.stdin"
  
    def _is_fixture_file_name(self, fixture_name):

        suffix_l = len(self.fixture_conf_file_suffix)
        if len(fixture_name) > suffix_l and \
            fixture_name[-suffix_l:] == self.fixture_conf_file_suffix:

            return True

        else:

            return False

    def get_fixture_name(self, fixture_path):

        fixture = fixture_path.split(os.sep)[-1]
        if len(fixture) > len(self.fixture_conf_file_suffix):
            if fixture[-len(self.fixture_conf_file_suffix):] == \
                self.fixture_conf_file_suffix:

                fixture = fixture[:-len(self.fixture_conf_file_suffix)]

        return fixture.capitalize().replace("_", " ")

    def get_scanner_path_name(self, scanner):

        return scanner.lower().replace(" ", "_")

    def get_fixture_path(self, fixture_name, conf_file=True, own_path=None,
            only_name=False):

        fixture_name = fixture_name.lower().replace(" ", "_")

        if self._is_fixture_file_name(fixture_name):
            fixture_name = fixture_name[:-len(self.fixture_conf_file_suffix)]

        if only_name:
            return fixture_name

        if own_path is not None:
            if conf_file:
                f_pattern = self.fixture_conf_file_rel_pattern
            else:
                f_pattern = self.fixture_image_file_rel_pattern

            if own_path == "":
                return f_pattern.format(fixture_name)
            else:
                return os.sep.join((own_path, f_pattern.format(fixture_name)))

        if conf_file:
            return self.fixture_conf_file_pattern.format(fixture_name)
        else:
            return self.fixture_image_file_pattern.format(fixture_name)

    def get_new_log_analysis(self):

        f = (self.log_analysis_out.format(
            self._last_analysis_log_index),
            self.log_analysis_err.format(
            self._last_analysis_log_index))

        self._last_analysis_log_index += 1

        #HACK, could cause conflict... but don't want overflow of logs
        if self._last_analysis_log_index > 20:
            self._last_analysis_log_index = 0

        return f
