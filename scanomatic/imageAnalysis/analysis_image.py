"""The module hold the object that coordinates plates"""
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
import matplotlib.image as plt_img
import numpy as np

#
# SCANNOMATIC LIBRARIES
#

import grid_array
import first_pass_image
import support
import scanomatic.io.paths as paths
import scanomatic.io.app_config as app_config_module

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
    def __init__(self, analysis_model, scanning_model):

        self._analysis_model = analysis_model
        self._scanning_model = scanning_model

        self._im_loaded = False

        self._ref_plate_d1 = None
        self._ref_plate_d2 = None

        self._paths = paths.Paths()

        #APP CONFIG
        self._app_config = app_config_module.Config()


        #Fixture setting is used for pinning history in the arrays
        if fixture_name is None:
            fixture_name = self._paths.experiment_local_fixturename
            fixture_directory = self._file_path_base
        else:
            fixture_name = self._paths.get_fixture_path(fixture_name, only_name=True)
            fixture_directory = None

        self.fixture = first_pass_image.Image(
            fixture_name,
            fixture_directory=fixture_directory,
            appConfig=self._app_config)

        self.im = None

        self.set_pinning_matrices(pinning_matrices)

    def __getitem__(self, key):

        return self._grid_arrays[key]

    def get_file_base_dir(self):

        return self._file_path_base

    def set_pinning_matrices(self, pinning_matrices):

        self.R = []
        self.features = []
        self._grid_arrays = []
        self._pinning_matrices = pinning_matrices

        for a in xrange(len(pinning_matrices)):

            if pinning_matrices[a] is not None:

                self._grid_arrays.append(grid_array.Grid_Array(
                    self, (a,),
                    pinning_matrices[a], visual=self.visual,
                    suppress_analysis=self.suppress_analysis,
                    grid_array_settings=self.grid_array_settings,
                    gridding_settings=self.gridding_settings,
                    grid_cell_settings=self.grid_cell_settings))

                self.features.append(None)
                self.R.append(None)

        if len(pinning_matrices) > len(self._grid_arrays):

            """
            self._logger.info(
                "Analysis will run on {0} plates out of {1}".format(
                len(self._grid_arrays), len(pinning_matrices)))
            """

    '''
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

                    self._logger.error('Failed to set manual grid "+ \
                        "adjustments to {0}, plate non-existent'.format(k))
    '''

    def set_grid(self, image_model, save_name=None):

        if save_name is None:
             save_name=os.sep.join((self._analysis_model.output_directory, "grid___origin_plate_"))

        #self._logger.info("Setting grids from image {0}".format(im_path))
        self._im_path = im_path
        self.load_image()
        if self._im_loaded:

            if self._log_version < self._app_config.version_first_pass_change_1:
                scale_factor = 4.0
            else:
                scale_factor = 1.0

            for idGA in xrange(len(self._grid_arrays)):

                """
                self._logger.info("Setting grid on plate {0}".format(
                    grid_array))
                """
                im = self.get_im_section(plate_positions[idGA], scale_factor)

                if self._grid_correction is None:
                    self._grid_arrays[idGA].set_grid(
                        im, save_name=save_name,
                        grid_correction=None)
                else:
                    self._grid_arrays[idGA].set_grid(
                        im, save_name=save_name,
                        grid_correction=self._grid_correction[idGA])

    def load_image(self, image_dict=None):

        try:

            self.im = plt_img.imread(self._im_path)
            self._im_loaded = True

        except:

            alt_path = os.path.join(self._file_path_base,
                                    os.path.basename(self._im_path))

            """
            self._logger.warning(
                "ANALYSIS IMAGE, Could not open image at " +
                "'{0}' trying in log-file directory ('{1}').".format(
                    self._im_path, alt_path))
            """
            try:

                self.im = plt_img.imread(alt_path)
                self._im_loaded = True
                self._im_path = alt_path

            except:

                #self._logger.warning("ANALYSIS IMAGE, No image found... sorry")
                self._im_loaded = False

        #This makes sure that the image is 'standing' and not a 'landscape'
        if (image_dict is not None and 'Image Shape' in image_dict and
                image_dict['Image Shape'] is not None):

            ref_shape = image_dict['Image Shape']

        else:

            ref_shape = (1, 0)

        if self.im is not None:
            self.im = support.get_first_rotated(
                self.im, ref_shape)

    def get_im_section(self, features, scale_factor=4.0, im=None,
                       run_insane=False):

        if self._im_loaded or im is not None:

            #SCALE AND ORDER BOUNDS
            F = np.round(np.array(features, dtype=np.float) *
                         scale_factor).astype(np.int)

            F.sort(axis=0)

            #SET AXIS ORDER DEPENDING ON VERSION
            if (self.fixture['version'] >=
                    self._app_config.version_first_pass_change_1):

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

                #Sections the area of the image referring to the plate
                plate_im = im[F[low, dim1]: F[high, dim1],
                              F[low, dim2]: F[high, dim2]]

                #Determines the shorter dimension
                short_dim = [p == min(plate_im.shape) for
                             p in plate_im.shape].index(True)

                #Causes the flipping along the short dimension
                slicer = [i != short_dim and slice(None, None, None) or
                          slice(None, None, -1) for i in range(plate_im.ndim)]

                return plate_im[slicer]

            else:

                raise Slice_Outside_Image(
                    "im {0} , slice {1}, scaled by {2} fixture {3} {4}".format(
                        im.shape,
                        np.s_[F[low, dim1]:F[high, dim1], F[low, dim2],
                              F[high, dim2]],
                        scale_factor,
                        self.fixture['name'],
                        self.fixture['version']))

    def _set_current_grid_move(self, d1, d2):

        self._grid_corrections = np.array((d1, d2))

    def _get_slice_sanity_check(self, im, d1=None, d2=None):

        if ((float(im.shape[0]) / im.shape[1] > 1) !=
                (float(d1) / d2 > 1)):

            s = "Current shape is {0} x {1}.".format(d1, d2)
            s += " Image is {0} x {1}.".format(im.shape[0], im.shape[1])
            raise Slice_Error(s)

            return False

        return True

    def get_analysis(self, image_model):

        #
        #   LOAD IMAGE
        #
        if im_path is not None:

            self._im_path = im_path
            self.load_image(image_dict=image_dict)

        #
        #   VERIFY IMAGE FORMAT
        #
        if self._im_loaded is True:

            if self.im.ndim == 3:
                #Makes image grayscale balancing colors' effect and
                #trashing the alpha-channel (if any)
                self.im = np.dot(self.im[..., :3], [0.299, 0.587, 0.144])
                #self._logger.warning("Color image got converted to grayscale")

        else:
            #self._logger.error("Failed to load image, all methods exhausted")
            return None

        #
        #   CONFIG FILE COMPATIBILITY, COORDINATE VALUE SCALINGS
        #

        if self._log_version < self._app_config.version_first_pass_change_1:
            scale_factor = 4.0
        else:
            scale_factor = 1.0

        if grayscaleSource is None:
            #self._logger.error("Grayscale sources can't be None")
            return None

        if grayscaleTarget is None:
            grayscaleTarget = self._grayscaleTarget
            if grayscaleTarget is None:
                #self._logger.error("Grayscale targets not inited correctly")
                pass

        for plateIndex in range(len(features)):

            im = self.get_im_section(features[plateIndex], scale_factor)
            gridArray = self._grid_arrays[plateIndex]
            gridArray.doAnalysis(
                im,
                grayscaleSource=grayscaleSource,
                grayscaleTarget=grayscaleTarget,
                watch_colony=watch_colony,
                save_grid_name=save_grid_name,
                identifier_time=identifier_time,
                grid_correction=self._grid_corrections)

            self.features[plateIndex] = gridArray.features
            self.R[plateIndex] = gridArray.R

        if watch_colony is not None:

            watchPlate = self._grid_arrays[watch_colony[0]]
            self.watch_grid_size = watchPlate._grid_cell_size

            self.watch_source = watchPlate.watch_source
            self.watch_blob = watchPlate.watch_blob

            self.watch_results = watchPlate.watch_results

        """
        for handler in self._logger.handlers:
            handler.flush()
        """

        return self.features

    def _get_grid_image_name(self, image_model):

        if image_model.index in self._analysis_job.grid_images:
            return os.sep.join((
                self._analysis_job.output_directory,
                "grid__time_index_{0}_plate_".format(str(self._iteration_index).zfill(4))))
        return None