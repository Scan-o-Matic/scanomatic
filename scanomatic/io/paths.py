"""Resource Paths"""
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
import re
import md5
import random

#
# INTERNAL DEPENDENCIES
#

import logger
import scanomatic.faulty_reference as faulty_reference

#
# EXCEPTIONS
#


class Invalid_Root(Exception):
    pass

#
# CLASSES
#


class Paths(object):

    def __init__(self, *args):

        self._logger = logger.Logger("Paths Class")
        if (len(args) > 0):
            self._logger.warning(
                "Some class instantiated a Paths object wit parameters." +
                " They are ignorded as this is no longer valid")

        self.root = os.path.join(os.path.expanduser("~"), ".scan-o-matic")

        if os.path.isdir(self.root) is False:
            raise Invalid_Root(self.root)

        self.config = os.path.join(self.root, "config")
        self.fixtures = os.path.join(self.config, "fixtures")
        self.images = os.path.join(self.root, "images")

        #INSTALL
        self.desktop_file = "scan-o-matic.desktop"
        self.desktop_file_path = os.path.join(
            self.config, self.desktop_file)
        self.install_filezilla = os.path.join(
            self.config, "install_filezilla.sh")

        #RUN-files
        self.scanomatic = "scan_o_matic"
        self.analysis = "scan-o-matic_analysis"
        self.experiment = "scan-o-matic_experiment"
        self.make_project = "scan-o-matic_make_project"
        self.revive = "scan-o-matic_relauncher"
        self.install_autostart = "scan-o-matic_autostart"

        #CONFIG
        self.config_main_app = os.path.join(self.config, 'main.config')
        self.config_mac = os.path.join(self.config, 'mac_address.config')
        self.config_rpc = os.path.join(self.config, 'rpc.config')
        self._config_rpc_admin = None

        #IMAGES
        self.marker = os.path.join(self.images, "orientation_marker_150dpi.png")
        self.martin = os.path.join(self.images, "martin3.png")
        self.logo = os.path.join(self.images, "scan-o-matic.png")

        #FIXTURE_FILES
        self.fixture_conf_file_suffix = ".config"
        self.fixture_conf_file_rel_pattern = "{0}" + \
            self.fixture_conf_file_suffix
        self.fixture_image_file_rel_pattern = "{0}.npy"
        self.fixture_conf_file_pattern = os.path.join(
            self.fixtures, self.fixture_conf_file_rel_pattern)
        self.fixture_image_file_pattern = os.path.join(
            self.fixtures, self.fixture_image_file_rel_pattern)
        self.fixture_tmp_scan_image = \
            self.fixture_image_file_pattern.format(".tmp")

        #LOG
        self.log = os.path.join(self.root, "logs")
        self.log_scanner_out = os.path.join(self.log, "scanner_{0}.stdout")
        self.log_scanner_err = os.path.join(self.log, "scanner_{0}.stderr")
        self.log_main_out = os.path.join(self.log, "main.stdout")
        self.log_main_err = os.path.join(self.log, "main.stderr")
        #self._last_analysis_log_index = 0
        self.log_rebuild_in = os.path.join(self.log, "rebuild_{0}.stdin")
        self.log_rebuild_out = os.path.join(self.log, "rebuild_{0}.stdout")
        self.log_rebuild_err = os.path.join(self.log, "rebuild_{0}.stderr")
        self.log_analysis_in = os.path.join(self.log, "analysis_{0}.stdin")
        self.log_analysis_out = os.path.join(self.log, "analysis_{0}.stdout")
        self.log_analysis_err = os.path.join(self.log, "analysis_{0}.stderr")
        self.log_relaunch = os.path.join(self.log, "relaunch.log")
        self.log_project_progress = os.path.join(self.log, "progress.projects")

        #EXPERIMENT
        self.experiment_root = os.path.join(os.path.expanduser("~"), "Documents")
        self.experiment_scan_image_pattern = "{0}_{1}_{2:.4f}.tiff"
        self.experiment_analysis_relative_path = "analysis"
        self.experiment_analysis_file_name = "analysis.log"
        self.experiment_rebuild_instructions = "rebuild.instructions"

        #ANALSYS FILES
        self.analysis_polynomial = os.path.join(
            self.config, "calibration.polynomials")
        self.analysis_calibration_data = os.path.join(
            self.config, "calibration.data")
        self.analysis_graycsales = os.path.join(
            self.config, "grayscales.cfg")

        self.analysis_run_log = 'analysis.run'

        self.experiment_first_pass_analysis_relative = "{0}.1_pass.analysis"
        self.experiment_first_pass_log_relative = ".1_pass.log"
        self.experiment_local_fixturename = \
            self.fixture_conf_file_rel_pattern.format("fixture")
        self.experiment_grid_image_pattern = "grid___origin_plate_{0}.svg"
        self.experiment_grid_error_image = "_no_grid_{0}.npy"

        #LOCK FILES
        self.lock_root = os.path.join(self.root, 'locks')
        self.lock_power_up_new_scanner = os.path.join(
            self.lock_root, "new_scanner.lock")
        self.lock_scanner_pattern = os.path.join(
            self.lock_root, "scanner.{0}.lock")
        self.lock_scanner_addresses = os.path.join(
            self.lock_root, "addresses.lock")

        #EXPERIMENT FILE PIPE
        self.experiment_stdin = os.path.join(
            self.lock_root, "scanner.{0}.stdin")

    @property
    def config_rpc_admin(self):

        if self._config_rpc_admin is not None:
            return self._config_rpc_admin

        path = os.path.join(self.config, 'rpc.admin')
        if (os.path.isfile(path)):
            fh = open(path, 'r')
            admin = fh.read().strip()
            fh.close()
        else:
            admin = md5.new(str(random.random())).hexdigest()
            fh = open(path, 'w')
            fh.write(admin)
            fh.close()

        return admin

    @property
    def src(self):

        return faulty_reference.FaultyReference("src", base=str(self))

    def _is_fixture_file_name(self, fixture_name):

        suffix_l = len(self.fixture_conf_file_suffix)
        if (len(fixture_name) > suffix_l and
                fixture_name[-suffix_l:] ==
                self.fixture_conf_file_suffix):

            return True

        else:

            return False

    def get_fixture_name(self, fixture_path):

        fixture = os.path.basename(fixture_path)
        if len(fixture) > len(self.fixture_conf_file_suffix):
            if fixture[-len(self.fixture_conf_file_suffix):] == \
                    self.fixture_conf_file_suffix:

                fixture = fixture[:-len(self.fixture_conf_file_suffix)]

        return fixture.capitalize().replace("_", " ")

    def get_scanner_path_name(self, scanner):

        return scanner.lower().replace(" ", "_")

    def get_scanner_index(self, scanner_path):

        candidates = map(int, re.findall(r"\d+", scanner_path))
        if len(candidates) > 0:
            return candidates[-1]
        else:
            return None

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

                f = f_pattern.format(fixture_name)
                if os.path.isfile(f):
                    return f
            else:
                f = os.path.join(own_path, f_pattern.format(fixture_name))
                if os.path.isfile(f):
                    return f

        if conf_file:
            return self.fixture_conf_file_pattern.format(fixture_name)
        else:
            return self.fixture_image_file_pattern.format(fixture_name)
