"""The module hold the object that coordinates plates"""
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
import matplotlib.image as plt_img
import logging
import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import analysis_grid_array
import resource_fixture_image 
import resource_path
import resource_app_config

#
# EXCEPTIONS
#

class Slice_Outside_Image(Exception): pass

#
# CLASS Project_Image
#


class Project_Image():
    def __init__(self, pinning_matrices, im_path=None, plate_positions=None,
        animate=False, file_path_base="", fixture_name=None,
        p_uuid=None, logger=None, verbose=False, visual=False,
        suppress_analysis=False,
        grid_array_settings=None, gridding_settings=None,
        grid_cell_settings=None, log_version=0):

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger('Scan-o-Matic Analysis')

        self.p_uuid = p_uuid
        self._log_version = log_version

        self._im_path = im_path
        self._im_loaded = False

        self._plate_positions = plate_positions
        self._pinning_matrices = pinning_matrices

        self.verbose = verbose
        self.visual = visual
        self.suppress_analysis = suppress_analysis

        self.grid_array_settings = grid_array_settings
        self.gridding_settings = gridding_settings
        self.grid_cell_settings = grid_cell_settings

        #PATHS
        script_path_root = os.path.dirname(os.path.abspath(__file__))
        scannomatic_root = os.sep.join(script_path_root.split(os.sep)[:-1])
        self._paths = resource_path.Paths(root=scannomatic_root)
        self._file_path_base = file_path_base

        #APP CONFIG
        self._config = resource_app_config.Config(self._paths)

        #Fixture setting is used for pinning history in the arrays
        if fixture_name is None:
            fixture_name = self._paths.experiment_local_fixturename
            fixture_directory = self._file_path_base  
        else:
            fixture_name = self._paths.get_fixture_path(fixture_name, only_name=True) 
            fixture_directory = None

        self.fixture = resource_fixture_image.Fixture_Image(
                fixture_name,
                fixture_directory=fixture_directory,
                logger=self.logger
                )

        self.logger.info("Fixture is {0}, version {1}".format(
            self.fixture['name'], self.fixture['version']))

        self.im = None

        self.gs_indices = np.asarray([82, 78, 74, 70, 66, 62, 58, 54, 50, 46,
                            42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 4, 2, 0])

        self._timestamp = None
        self.set_pinning_matrices(pinning_matrices)

    def set_pinning_matrices(self, pinning_matrices):

        self.R = []
        self.features = []
        self._grid_arrays = []
        self._pinning_matrices = pinning_matrices

        for a in xrange(len(pinning_matrices)):

            if pinning_matrices[a] is not None:

                self._grid_arrays.append(analysis_grid_array.Grid_Array(
                        self, (a,),
                        pinning_matrices[a], verbose=self.verbose,
                        visual=self.visual,
                        suppress_analysis=self.suppress_analysis,
                        grid_array_settings=self.grid_array_settings,
                        gridding_settings=self.gridding_settings,
                        grid_cell_settings=self.grid_cell_settings))

                self.features.append(None)
                self.R.append(None)

        if len(pinning_matrices) > len(self._grid_arrays):

            self.logger.info('Analysis will run on " + \
                    "{0} plates out of {1}'.format(
                    len(self._grid_arrays), len(pinning_matrices)))

    def set_manual_ideal_grids(self, grid_adjustments):
        """Overrides grid detection with a specified grid supplied in grid
        adjustments

        @param grid_adjustments:    A dictionary of pinning grids with plate
                                    numbers as keys and items being tuples of
                                    row and column position lists.
        """

        for k in grid_adjustments.keys():

            if self._pinning_matrices[k] is not None:

                try:

                    self._grid_arrays[k].set_manual_ideal_grid(grid_adjustments[k])

                except IndexError:

                    self.logger.error('Failed to set manual grid "+ \
                        "adjustments to {0}, plate non-existent'.format(k))

    def set_grid(self, im_path, plate_positions, save_name=None):

        self.logger.info("Setting grids from image {0}".format(im_path))
        self._im_path = im_path
        self.load_image()
        if self._im_loaded:

            if self._log_version < self._config.version_first_pass_change_1:
                scale_factor = 4.0
            else:
                scale_factor = 1.0

            for grid_array in xrange(len(self._grid_arrays)):

                self.logger.info("Setting grid on plate {0}".format(
                    grid_array))

                im = self.get_im_section(plate_positions[grid_array], scale_factor)

                self._grid_arrays[grid_array].set_grid(im, save_name=save_name)

    def load_image(self):

        try:

            self.im = plt_img.imread(self._im_path)
            self._im_loaded = True

        except:

            alt_path = os.sep.join((self._file_path_base,
                    self._im_path.split(os.sep)[-1]))

            self.logger.warning("ANALYSIS IMAGE, Could not open image at " + \
                    "'{0}' trying in log-file directory ('{1}').".format(
                    self._im_path, alt_path))

            try:

                self.im = plt_img.imread(alt_path)
                self._im_loaded = True

            except:

                self.logger.warning("ANALYSIS IMAGE, No image found... sorry")
                self._im_loaded = False

    def get_plate(self, plate_index):

        if -1 < plate_index < len(self._grid_arrays):

            return self._grid_arrays[plate_index]

        else:

            self.logger.warning("ANALYSIS IMAGE: Plate " + \
                        "{0} outside expected range (0 - {1}).".format(
                        plate_index, len(self._grid_arrays)))

            return None

    def get_im_section(self, features, scale_factor=4.0):

        if self._im_loaded:

            x0 = round(features[0][0] * scale_factor)
            x1 = round(features[1][0] * scale_factor)

            if x0 < x1:

                upper = x0
                lower = x1

            else:

                upper = x1
                lower = x0

            y0 = round(features[0][1] * scale_factor)
            y1 = round(features[1][1] * scale_factor)

            if y0 < y1:

                left = y0
                right = y1

            else:

                left = y1
                right = y0

            if self.fixture['version'] >= self._config.version_first_pass_change_1:
                if self._get_slice_sanity_check(d1=right, d2=lower):
                    return self.im[left: right, upper:lower]
                else:
                    raise Slice_Outside_Image(
                        "im {0} , slice {1}, scaled by {2} fixture {3} {4}".format(
                        self.im.shape,
                        np.s_[left:right, upper:lower],
                        scale_factor,
                        self.fixture['name'],
                        self.fixture['version']))

            else:
                if self._get_slice_sanity_check(d1=lower, d2=right):
                    return self.im[upper: lower, left: right]
                else:
                    raise Slice_Outside_Image(
                        "im {0} , slice {1}, scaled by {2}, fixture {3} {4}".format(
                        self.im.shape,
                        np.s_[upper:lower, left:right],
                        scale_factor,
                        self.fixture['name'],
                        self.fixture['version']))

        else:
            return None

    def _get_slice_sanity_check(self, d1=None, d2=None):

        if d1 is not None and self.im.shape[0] < d1:
            return False

        if d2 is not None and self.im.shape[1] < d2:
            return False

        return True

    def get_analysis(self, im_path, features, grayscale_values,
            watch_colony=None, save_grid_name=None,
            grid_lock=False, identifier_time=None, timestamp=None,
            grayscale_indices=None):

        """
            @param im_path: An path to an image

            @param features: A list of pinning grids to look for

            @param grayscale_values : An array of the grayscale pixelvalues,
            if submittet gs_fit is disregarded

            @param use_fallback : Causes fallback detection to be used.

            @param watch_colony : A particular colony to gather information
            about.

            @param suppress_other : If only the watched colony should be
            analysed

            @param save_grid_name : A custom name for the saved image, if none
            is submitted, it will be grid.png in current directory.

            @param grid_lock : Default False, if true, the grid will only be
            gotten once and then reused all way through.

            @param identifier_time : A time index to update the identifier with

            The function returns two arrays, one per dimension, of the
            positions of the spikes and a quality index
        """

        if im_path != None:

            self._im_path = im_path
            self.load_image()

        if self._im_loaded == True:

            if len(self.im.shape) > 2:

                self.im = self.im[:, :, 0]

        else:

            return None

        self._timestamp = timestamp

        if len(grayscale_values) > 3:

            gs_values = np.array(grayscale_values)

            if grayscale_indices is None:

                gs_indices = self.gs_indices

            else:

                gs_indices = np.array(grayscale_indices)
                self.gs_indices = gs_indices

            gs_fit = np.polyfit(gs_indices, gs_values, 3)

        else:

            gs_fit = None

        self.logger.debug("ANALYSIS produced gs-coefficients" + \
                    " {0} ".format(gs_fit))

        if self._log_version < self._config.version_first_pass_change_1:
            scale_factor = 4.0
        else:
            scale_factor = 1.0

        if gs_fit is not None:

            z3_deriv_coeffs = np.array(gs_fit[: -1]) * \
                        np.arange(gs_fit.shape[0] - 1, 0, -1)

            z3_deriv = np.array(map(lambda x: (z3_deriv_coeffs * np.power(x,
                np.arange(z3_deriv_coeffs.shape[0], 0, -1))).sum(), range(87)))

            if (z3_deriv > 0).any() and (z3_deriv < 0).any():

                self.logger.warning("ANALYSIS of grayscale seems dubious" + \
                                " as coefficients don't have the same sign")

                gs_fit = None

        if gs_fit is None:

            return None

        for grid_array in xrange(len(self._grid_arrays)):

            im = self.get_im_section(features[grid_array], scale_factor)

            self._grid_arrays[grid_array].get_analysis(
                    im,
                    gs_values=gs_values,
                    watch_colony=watch_colony,
                    save_grid_name=save_grid_name,
                    identifier_time=identifier_time)

            self.features[grid_array] = self._grid_arrays[grid_array]._features
            self.R[grid_array] = self._grid_arrays[grid_array].R

        if watch_colony != None:

            self.watch_grid_size = \
                    self._grid_arrays[watch_colony[0]]._grid_cell_size

            self.watch_source = self._grid_arrays[watch_colony[0]].watch_source
            self.watch_scaled = self._grid_arrays[watch_colony[0]].watch_scaled
            self.watch_blob = self._grid_arrays[watch_colony[0]].watch_blob

            self.watch_results = \
                    self._grid_arrays[watch_colony[0]].watch_results

        return self.features


