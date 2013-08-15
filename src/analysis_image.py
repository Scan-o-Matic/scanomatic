"""The module hold the object that coordinates plates"""
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
import matplotlib.image as plt_img
import logging
import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import analysis_grid_array
import resource_fixture_image
import resource_analysis_support
import resource_path
import resource_app_config

#
# EXCEPTIONS
#


class Slice_Outside_Image(Exception):
    pass


class Slice_Error(Exception):
    pass

#
# CLASS Project_Image
#


class Project_Image():
    def __init__(
            self, pinning_matrices, im_path=None, plate_positions=None,
            animate=False, file_path_base="", fixture_name=None,
            p_uuid=None, logger=None, verbose=False, visual=False,
            suppress_analysis=False,
            grid_array_settings=None, gridding_settings=None,
            grid_cell_settings=None, log_version=0, paths=None,
            app_config=None):

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
        self._ref_plate_d1 = None
        self._ref_plate_d2 = None
        self._grid_corrections = None

        self.verbose = verbose
        self.visual = visual
        self.suppress_analysis = suppress_analysis

        self.grid_array_settings = grid_array_settings
        self.gridding_settings = gridding_settings
        self.grid_cell_settings = grid_cell_settings

        #PATHS
        if paths is None:
            self._paths = resource_path.Paths(src_path=__file__)
        else:
            self._paths = paths

        self._file_path_base = file_path_base

        #APP CONFIG
        if app_config is None:
            self._config = resource_app_config.Config(paths=self._paths)
        else:
            self._config = app_config

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
            logger=self.logger, paths=self._paths,
            app_config=self._config)

        self.logger.info("Fixture is {0}, version {1}".format(
            self.fixture['name'], self.fixture['version']))

        self.im = None

        self.gs_indices = np.asarray(
            [82, 78, 74, 70, 66, 62, 58, 54, 50, 46,
             42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 4, 2, 0])

        self._timestamp = None
        self.set_pinning_matrices(pinning_matrices)

    def get_file_base_dir(self):

        return self._file_path_base

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

            self.logger.info(
                "Analysis will run on {0} plates out of {1}".format(
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

                self._grid_arrays[grid_array].set_grid(
                    im, save_name=save_name,
                    grid_correction=self._grid_corrections)

    def load_image(self, image_dict=None):

        try:

            self.im = plt_img.imread(self._im_path)
            self._im_loaded = True

        except:

            alt_path = os.path.join(self._file_path_base,
                                    os.path.basename(self._im_path))

            self.logger.warning(
                "ANALYSIS IMAGE, Could not open image at " +
                "'{0}' trying in log-file directory ('{1}').".format(
                    self._im_path, alt_path))

            try:

                self.im = plt_img.imread(alt_path)
                self._im_loaded = True

            except:

                self.logger.warning("ANALYSIS IMAGE, No image found... sorry")
                self._im_loaded = False

        #This makes sure that the image is 'standing' and not a 'landscape'
        if (image_dict is not None and 'Image Shape' in image_dict and
                image_dict['Image Shape'] is not None):

            ref_shape = image_dict['Image Shape']

        else:

            ref_shape = (1, 0)

        if self.im is not None:
            self.im = resource_analysis_support.get_first_rotated(
                self.im, ref_shape)

    def get_plate(self, plate_index):

        if -1 < plate_index < len(self._grid_arrays):

            return self._grid_arrays[plate_index]

        else:

            self.logger.warning(
                "ANALYSIS IMAGE: Plate " +
                "{0} outside expected range (0 - {1}).".format(
                plate_index, len(self._grid_arrays)))

            return None

    def get_im_section(self, features, scale_factor=4.0, im=None,
                       run_insane=False):

        if self._im_loaded or im is not None:

            #SCALE AND ORDER BOUNDS
            F = np.round(np.array(features, dtype=np.float) *
                         scale_factor).astype(np.int)

            F.sort(axis=0)

            #SET AXIS ORDER DEPENDING ON VERSION
            if (self.fixture['version'] >=
                    self._config.version_first_pass_change_1):

                dim1 = 1
                dim2 = 0

            else:

                dim1 = 0
                dim2 = 1

            #TAKE LOADED IM IF NON SUPPLIED
            if im is None:
                im = self.im

            #CORRECT FOR IM OUT OF BOUNDS
            low = 0
            high = 1
            d1_correction = 0
            d2_correction = 0

            if F[low, dim1] < 0:
                #upper_correction = F[0, dim1]
                F[low, dim1] = 0
            if F[low, dim2] < 0:
                #upper_correction = F[0, dim2]
                F[low, dim2] = 0

            if F[high, dim1] > im.shape[0]:
                F[high, dim1] = im.shape[0]
            if F[high, dim2] > im.shape[1]:
                F[high, dim2] = im.shape[1]

            #RECORD LOW VALUE CORRECTION ON EITHER DIM (THIS MEANS GRID
            #NEEDS TO BE OFFSET
            self._set_current_grid_move(d1=d1_correction, d2=d2_correction)

            #CHECK SO THAT PLATE SHAPE IS AGREEING WITH IM SHAPE
            #(MUST HAVE THE SAME ORIENTATION)
            if self._get_slice_sanity_check(
                    im,
                    d1=F[high, dim1] - F[low, dim1],
                    d2=F[high, dim2] - F[low, dim2]) or run_insane:

                return im[F[low, dim1]: F[high, dim1], F[low, dim2]: F[high, dim2]]

            else:

                raise Slice_Outside_Image(
                    "im {0} , slice {1}, scaled by {2} fixture {3} {4}".format(
                        im.shape,
                        np.s_[F[low, dim1]:F[high, dim1], F[low, dim2],
                              F[high, dim2]],
                        scale_factor,
                        self.fixture['name'],
                        self.fixture['version']))

            """
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

            upper_correction = 0
            left_correction = 0

            if upper < 0:
                upper_correction = upper
                upper = 0
            if left < 0:
                left_correction = left
                left = 0

            if self.fixture['version'] >= self._config.version_first_pass_change_1:

                if lower > im.shape[1]:
                    right = im.shape[1]
                if right > im.shape[0]:
                    right = im.shape[0]

                self._set_current_grid_move(d1=left_correction, d2=upper_correction)

                if self._get_slice_sanity_check(d1=right-left, d2=lower-upper) or run_insane:
                    return im[left: right, upper:lower]
                else:
                    raise Slice_Outside_Image(
                        "im {0} , slice {1}, scaled by {2} fixture {3} {4}".format(
                        im.shape,
                        np.s_[left:right, upper:lower],
                        scale_factor,
                        self.fixture['name'],
                        self.fixture['version']))

            else:

                if lower > im.shape[0]:
                    right = im.shape[0]
                if right > im.shape[1]:
                    right = im.shape[1]

                self._set_current_grid_move(d1=left_correction, d2=upper_correction)

                if self._get_slice_sanity_check(d1=lower-upper, d2=right-left) or run_insane:
                    return im[upper: lower, left: right]
                else:
                    raise Slice_Outside_Image(
                        "im {0} , slice {1}, scaled by {2}, fixture {3} {4}".format(
                        im.shape,
                        np.s_[upper:lower, left:right],
                        scale_factor,
                        self.fixture['name'],
                        self.fixture['version']))

        else:
            return None

        """

    def _set_current_grid_move(self, d1, d2):

        self._grid_corrections = np.array((d1, d2))

    def _get_slice_sanity_check(self, im, d1=None, d2=None):

        if ((float(im.shape[0]) / im.shape[1] > 0) !=
                (float(d1 / d2) > 0)):

            s = "Current shape is {0} x {1}.".format(d1, d2)
            s += " Image is {0} x {1}.".format(self._ref_plate_d1, self._ref_plate_d2)
            raise Slice_Error(s)

            return False

        return True

    def get_analysis(
            self, im_path, features, grayscale_values,
            watch_colony=None, save_grid_name=None,
            grid_lock=False, identifier_time=None, timestamp=None,
            grayscale_indices=None, image_dict=None):

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

        if im_path is not None:

            self._im_path = im_path
            self.load_image(image_dict=image_dict)

        if self._im_loaded is True:

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

        self.logger.debug("ANALYSIS produced gs-coefficients {0} ".format(
            gs_fit))

        if self._log_version < self._config.version_first_pass_change_1:
            scale_factor = 4.0
        else:
            scale_factor = 1.0

        if gs_fit is not None:

            z3_deriv_coeffs = (np.array(gs_fit[: -1]) *
                               np.arange(gs_fit.shape[0] - 1, 0, -1))

            z3_deriv = np.array(
                map(lambda x: (z3_deriv_coeffs * np.power(
                    x, np.arange(z3_deriv_coeffs.shape[0], 0, -1))).sum(),
                    range(87)))

            z3_deriv = z3_deriv[z3_deriv != 0]

            if (z3_deriv > 0).any() and (z3_deriv < 0).any():

                self.logger.warning(
                    "ANALYSIS of grayscale seems dubious" +
                    " check the coefficients: {0}".format(
                        gs_fit))

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
                identifier_time=identifier_time,
                grid_correction=self._grid_corrections)

            self.features[grid_array] = self._grid_arrays[grid_array]._features
            self.R[grid_array] = self._grid_arrays[grid_array].R

        if watch_colony is not None:

            self.watch_grid_size = \
                self._grid_arrays[watch_colony[0]]._grid_cell_size

            self.watch_source = self._grid_arrays[watch_colony[0]].watch_source
            self.watch_blob = self._grid_arrays[watch_colony[0]].watch_blob

            self.watch_results = \
                self._grid_arrays[watch_colony[0]].watch_results

        return self.features
