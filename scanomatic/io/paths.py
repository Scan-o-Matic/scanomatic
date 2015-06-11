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

#
# INTERNAL DEPENDENCIES
#

import logger
import scanomatic.faulty_reference as faulty_reference
from scanomatic.generics.singleton import SingeltonOneInit

#
# EXCEPTIONS
#


class InvalidRoot(Exception):
    pass

#
# CLASSES
#


class Paths(SingeltonOneInit):

    def __one_init__(self, *args):

        self._logger = logger.Logger("Paths Class")
        if len(args) > 0:
            self._logger.warning(
                "Some class instantiated a Paths object wit parameters." +
                " They are ignorded as this is no longer valid")

        self.root = os.path.join(os.path.expanduser("~"), ".scan-o-matic")

        if os.path.isdir(self.root) is False:
            raise InvalidRoot(self.root)

        self.config = os.path.join(self.root, "config")
        self.fixtures = os.path.join(self.config, "fixtures")
        self.images = os.path.join(self.root, "images")

        self.desktop_file = "scan-o-matic.desktop"
        self.desktop_file_path = os.path.join(
            self.config, self.desktop_file)
        self.install_filezilla = os.path.join(
            self.config, "install_filezilla.sh")

        self.scanomatic = "scan_o_matic"
        self.analysis = "scan-o-matic_analysis"
        self.experiment = "scan-o-matic_experiment"
        self.make_project = "scan-o-matic_compile_project"
        self.revive = "scan-o-matic_relauncher"
        self.install_autostart = "scan-o-matic_autostart"

        self.config_main_app = os.path.join(self.config, 'main.config')
        self.config_mac = os.path.join(self.config, 'mac_address.config')
        self.config_rpc = os.path.join(self.config, 'rpc.config')
        self.config_rpc_admin = os.path.join(self.config, 'rpc.admin')
        self.config_scanners = os.path.join(self.config, 'scanners.config')

        self.rpc_queue = os.path.join(self.root, 'job_queue.cfg')
        self.rpc_jobs = os.path.join(self.root, 'jobs.cfg')
        self.rpc_scanner_status = os.path.join(self.root, 'scanner_status.cfg')

        self.ui_root = os.path.join(self.root, "ui_server")
        self.ui_css = os.path.join(self.ui_root, "style")
        self.ui_js = os.path.join(self.ui_root, "js")
        self.help_file = "help.html"
        self.fixture_file = "fixture.html"

        self.marker = os.path.join(self.images, "orientation_marker_150dpi.png")
        self.martin = os.path.join(self.images, "martin3.png")
        self.logo = os.path.join(self.images, "scan-o-matic.png")

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
        self.fixture_grid_history_pattern = "{0}.grid.history"

        self.log = os.path.join(self.root, "logs")
        self.log_scanner_out = os.path.join(self.log, "scanner_{0}.stdout")
        self.log_scanner_err = os.path.join(self.log, "scanner_{0}.stderr")
        self.log_main_out = os.path.join(self.log, "main.stdout")
        self.log_main_err = os.path.join(self.log, "main.stderr")

        self.log_relaunch = os.path.join(self.log, "relaunch.log")
        self.log_project_progress = os.path.join(self.log, "progress.projects")

        self.experiment_root = os.path.join(os.path.expanduser("~"), "Documents")
        self.experiment_scan_image_pattern = "{0}_{1}_{2:.4f}.tiff"
        self.experiment_analysis_relative_path = "analysis"
        self.experiment_analysis_file_name = "analysis.log"
        self.experiment_rebuild_instructions = "rebuild.instructions"

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

        self.phenotypes_raw_csv = "phenotypes_raw.csv"
        self.phenotypes_raw_npy = "phenotypes_raw.npy"
        self.phenotypes_filter = "phenotypes_filter.npy"
        self.phenotypes_input_data = "curves_raw.npy"
        self.phenotypes_input_smooth = "curves_smooth.npy"
        self.phenotypes_extraction_params = "phenotype_params.npy"
        self.phenotype_times = "phenotype_times.npy"

        self.image_analysis_img_data = "image_{0}_data.npy"
        self.image_analysis_time_series = "time_data.npy"

        self.project_settings_file_pattern = "{0}.project.settings"
        self.project_compilation_pattern = "{0}.project.compilation"
        self.project_compilation_instructions_pattern = "{0}.project.compilation.instructions"
        self.scan_project_file_pattern = "{0}.scan.instructions"

    def join(self, attr, *other):
        
        if hasattr(self, attr):
            return os.path.join(getattr(self, attr), *other)
        else:
            raise AttributeError("Unknown path attribute '{0}'".format(attr))

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

    def get_project_settings_path_from_scan_model(self, scan_model):

        return self.project_settings_file_pattern.format(
            os.path.join(scan_model.directory_containing_project, scan_model.project_name, scan_model.project_name))

    def get_project_compile_path_from_compile_model(self, compile_model):
        """

        :type compile_model: scanomatic.models.compile_project_model.CompileInstructionsModel
        :rtype : str
        """

        if os.path.isdir(compile_model.path):

            project_name = self.get_project_directory_name_from_path(compile_model.path)
            return self.project_compilation_pattern.format(os.path.join(compile_model.path, project_name))

        return compile_model.path

    def get_project_directory_name_from_path(self, path):

        return path.rstrip(os.sep).split(os.sep)[-1]

    def get_project_compile_instructions_path_from_compile_model(self, compile_model):

        project_name = self.get_project_directory_name_from_path(compile_model.path)
        return self.project_compilation_instructions_pattern.format(os.path.join(compile_model.path, project_name))

    @staticmethod
    def get_scanner_path_name(scanner):

        return scanner.lower().replace(" ", "_")

    @staticmethod
    def get_scanner_index(scanner_path):

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
