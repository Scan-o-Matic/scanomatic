from __future__ import absolute_import

import os
import re

#
# INTERNAL DEPENDENCIES
#

from . import logger
import scanomatic
from scanomatic.generics.singleton import SingeltonOneInit

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

        self.root = os.path.abspath(
            os.environ.get("SCANOMATIC_DATA",
            os.path.join(os.path.expanduser("~"), ".scan-o-matic")))

        make_directory(self.root)

        self.config = os.path.join(self.root, "config")
        make_directory(self.config)
        self.fixtures = os.path.join(self.config, "fixtures")
        make_directory(self.fixtures)
        self.images = os.path.join(
            scanomatic.get_location(), "images")

        self.source_location_file = os.path.join(self.root, "source_location.txt")

        self.desktop_file = "scan-o-matic.desktop"
        self.desktop_file_path = os.path.join(
            self.config, self.desktop_file)

        self.config_main_app = os.path.join(self.config, 'main.config')
        self.config_mac = os.path.join(self.config, 'mac_address.config')
        self.config_rpc = os.path.join(self.config, 'rpc.config')
        self.config_rpc_admin = os.path.join(self.config, 'rpc.admin')
        self.config_scanners = os.path.join(self.config, 'scanners.config')

        self.rpc_queue = os.path.join(self.root, 'job_queue.cfg')
        self.rpc_jobs = os.path.join(self.root, 'jobs.cfg')
        self.rpc_scanner_status = os.path.join(self.root, 'scanner_status.cfg')

        self.ui_root = os.path.join(
            scanomatic.get_location(), "ui_server_data")
        self.ui_css = os.path.join(self.ui_root, "style")
        self.ui_js = os.path.join(self.ui_root, "js")
        self.ui_font = os.path.join(self.ui_root, "fonts")

        self.marker = os.path.join(self.images, "orientation_marker_150dpi.png")

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

        self.log = os.path.join(self.root, "logs")
        make_directory(self.log)
        self.log_ui_server = os.path.join(self.log, "ui_server.log")
        self.log_server = os.path.join(self.log, "server.log")

        self.experiment_scan_image_pattern = "{0}_{1}_{2:.4f}.tiff"
        self.experiment_analysis_relative_path = "analysis"

        self.analysis_grayscales = os.path.join(
            self.config, "grayscales.cfg")

        self.ccc_folder = os.path.join(self.config, "ccc")
        make_directory(self.ccc_folder)
        self.ccc_file_pattern = os.path.join(self.ccc_folder, "{0}.ccc")
        self.ccc_image_pattern = os.path.join(self.ccc_folder, "{0}.{1}.tiff")
        self.ccc_image_plate_slice_pattern = os.path.join(
            self.ccc_folder, "{0}.{1}.plate_{2}.npy")
        self.ccc_image_plate_transformed_slice_pattern = os.path.join(
            self.ccc_folder, "{0}.{1}.plate_{2}.transformed.npy")
        self.ccc_image_gs_slice_pattern = os.path.join(
            self.ccc_folder, "{0}.{1}.gs.npy")
        self.ccc_image_plate_grid_pattern = os.path.join(
            self.ccc_folder,
            "{0}.{1}.plate{2}.grid.npy")

        self.analysis_run_log = 'analysis.log'
        self.analysis_model_file = 'analysis.model'

        self.experiment_local_fixturename = \
            self.fixture_conf_file_rel_pattern.format("fixture")
        self.experiment_grid_image_pattern = "grid___origin_plate_{0}.svg"
        self.grid_pattern = "grid_plate___{0}.npy"
        self.grid_size_pattern = "grid_size___{0}.npy"
        self.experiment_grid_error_image = "_no_grid_{0}.npy"

        self.ui_server_phenotype_state_lock = "phenotypes_state.lock"
        self.phenotypes_csv_pattern = "phenotypes.{0}.plate_{1}.csv"
        self.phenotypes_raw_npy = "phenotypes_raw.npy"
        self.vector_phenotypes_raw = "phenotypes_vectors_raw.npy"
        self.vector_meta_phenotypes_raw = "phenotypes_meta_vector_raw.npy"
        self.normalized_phenotypes = "normalized_phenotypes.npy"
        self.phenotypes_filter = "phenotypes_filter.npy"
        self.phenotypes_reference_offsets = "phenotypes_reference_offsets.npy"
        self.phenotypes_filter_undo = "phenotypes_filter.undo.pickle"
        self.phenotypes_meta_data = "meta_data.pickle"
        self.phenotypes_meta_data_original_file_patern = "meta_data_{0}.{1}"
        self.phenotypes_input_data = "curves_raw.npy"
        self.phenotypes_input_smooth = "curves_smooth.npy"
        self.phenotypes_extraction_params = "phenotype_params.npy"
        self.phenotype_times = "phenotype_times.npy"

        self.phenotypes_extraction_log = "phenotypes.extraction.log"
        self.phenotypes_extraction_instructions = "phenotypes.extraction.instructions"

        self.image_analysis_img_data = "image_{0}_data.npy"
        self.image_analysis_time_series = "time_data.npy"

        self.project_compilation_from_scanning_pattern_old = "{0}.project.settings"
        self.project_compilation_from_scanning_pattern = "{0}.project.compilation.original"
        self.project_compilation_pattern = "{0}.project.compilation"
        self.project_compilation_instructions_pattern = "{0}.project.compilation.instructions"
        self.project_compilation_log_pattern = "{0}.project.compilation.log"

        self.scan_project_file_pattern = "{0}.scan.instructions"
        self.scan_log_file_pattern = "{0}.scan.log"

    def join(self, attr, *other):

        if hasattr(self, attr):
            return os.path.join(getattr(self, attr), *other)
        else:
            raise AttributeError("Unknown path attribute '{0}'".format(attr))

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

    def get_original_compilation_path_from_scan_model(self, scan_model):

        return self.project_compilation_from_scanning_pattern.format(
            os.path.join(scan_model.directory_containing_project, scan_model.project_name, scan_model.project_name))

    def get_project_compile_path_from_compile_model(self, compile_model):
        """

        :type compile_model: scanomatic.models.compile_project_model.CompileInstructionsModel
        :rtype : str
        """

        if os.path.isdir(compile_model.path):

            return self.project_compilation_pattern.format(
                self.get_project_directory_name_with_file_prefix_from_path(compile_model.path))

        return compile_model.path

    @staticmethod
    def get_project_directory_name_with_file_prefix_from_path(path):

        if os.path.isdir(path):
            dir_name = path
        else:
            dir_name = os.path.dirname(path)
        return os.path.join(dir_name, dir_name.rstrip(os.sep).split(os.sep)[-1])

    def get_project_compile_instructions_path_from_compile_model(self, compile_model):

        return self.get_project_compile_instructions_path_from_compilation_path(compile_model.path)

    def get_project_compile_instructions_path_from_compilation_path(self, path):

        return self.project_compilation_instructions_pattern.format(
            self.get_project_directory_name_with_file_prefix_from_path(path))

    def get_project_compile_log_path_from_compile_model(self, compile_model):

        return self.project_compilation_log_pattern.format(
            self.get_project_directory_name_with_file_prefix_from_path(compile_model.path))

    def get_scan_instructions_path_from_compile_instructions_path(self, path):

        return self.scan_project_file_pattern.format(self.get_project_directory_name_with_file_prefix_from_path(path))

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


def make_directory(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
