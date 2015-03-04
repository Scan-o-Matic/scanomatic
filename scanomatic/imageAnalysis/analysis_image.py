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
from enum import Enum
#
# SCANNOMATIC LIBRARIES
#

import grid_array
import first_pass_image
from scanomatic.io.paths import Paths
import scanomatic.io.app_config as app_config_module


#
# EXCEPTIONS
#


class Slice_Outside_Image(Exception):
    pass


class Slice_Error(Exception):
    pass

IMAGE_ROTATIONS = Enum("IMAGE_ROTATIONS", names=("Landscape", "Portrait", "None"))


#
# CLASS Project_Image
#


class Project_Image(object):

    def __init__(self, analysis_model, scanning_model):

        self._analysis_model = analysis_model
        self._scanning_model = scanning_model

        self.fixture = self._load_fixture()

        self._im_loaded = False
        self.im = None

        #APP CONFIG
        self._app_config = app_config_module.Config()


        self._grid_arrays = self._get_grid_arrays()
        self.features = [None] * (max(self._grid_arrays.keys()) + 1)

    def _load_fixture(self):

        paths = Paths()
        if self._analysis_model.use_local_fixture or not self._scanning_model.fixture_name:
            fixture_name = paths.experiment_local_fixturename
            fixture_directory = os.path.dirname(self._analysis_model.first_pass_file)
        else:
            fixture_name = paths.get_fixture_path(self._scanning_model.fixture_name, only_name=True)
            fixture_directory = None

        return first_pass_image.Image(
            fixture_name,
            fixture_directory=fixture_directory,
            appConfig=self._app_config)


    def __getitem__(self, key):

        return self._grid_arrays[key]

    def _get_grid_arrays(self):

        grid_arrays = {}

        for index, pinning in enumerate(self._analysis_model.pinning_matrices):

           if pinning and self._plate_is_analysed(index):

                grid_arrays[index] = grid_array.Grid_Array(index, pinning, self.fixture, self._analysis_model)

        return grid_arrays

    def _plate_is_analysed(self, index):

        return not self._analysis_model.suppress_non_focal or index == self._analysis_model.focus_position[0]

    def set_grid(self, image_model, save_name=None):

        if save_name is None:
             save_name=os.sep.join((self._analysis_model.output_directory, "grid___origin_plate_"))

        self.load_image(image_model.path)

        if self._im_loaded:

            if self._scanning_model.version < self._app_config.version_first_pass_change_1:
                scale_factor = 4.0
            else:
                scale_factor = 1.0

            for index, grid_array in enumerate(self._grid_arrays):

                plate_model = [plate_model for plate_model in image_model.plates if plate_model.index == index][0]

                im = self.get_im_section(plate_model, scale_factor)

                if self._analysis_model.grid.gridding_offsets is None:
                    self._grid_arrays[index].set_grid(
                        im, save_name=save_name,
                        grid_correction=None)
                else:
                    self._grid_arrays[index].set_grid(
                        im, save_name=save_name,
                        grid_correction=self._analysis_model.grid.gridding_offset[index])

    def load_image(self, path):

        try:

            self.im = plt_img.imread(path)
            self._im_loaded = True

        except:

            alt_path = os.path.join(os.path.dirname(self._analysis_model.first_pass_file),
                                    os.path.basename(path))

            try:

                self.im = plt_img.imread(alt_path)
                self._im_loaded = True

            except:

                self._im_loaded = False

        self.validate_rotation()
        self._convert_to_grayscale()

    def _convert_to_grayscale(self):
        if self.im.ndim == 3:
            self.im = np.dot(self.im[..., :3], [0.299, 0.587, 0.144])


    def validate_rotation(self):

        pass

    @property
    def orientation(self):

        if not self._im_loaded:
            return IMAGE_ROTATIONS.None
        elif self.im.shape[0] > self.im.shape[1]:
            return IMAGE_ROTATIONS.Landscape
        else:
            return IMAGE_ROTATIONS.Portait

    def get_im_section(self, plate_model, scale_factor=4.0, im=None):

        def _flip_axis(X, Y):

            return Y, X

        def _bound(bounds, X, Y):

            def bounds_check(bound, val):

                if 0 <= val < bound:
                    return  val
                elif val < 0:
                    return 0
                else:
                    return bound - 1

            return ((bounds_check(bounds[0], X[0]),
                     bounds_check(bounds[0], X[1])),
                    (bounds_check(bounds[1], Y[0]),
                     bounds_check(bounds[1], Y[1])))

        if not im:
            if self._im_loaded:
                im = self.im
            else:
                return

        X = sorted(plate_model.x1, plate_model.x2)
        Y = sorted(plate_model.y1, plate_model.y2)

        if (self.fixture['version'] >=
                self._app_config.version_first_pass_change_1):

            if self.orientation == IMAGE_ROTATIONS.Portait:
                X, Y = _flip_axis(X, Y)

        elif self.orientation == IMAGE_ROTATIONS.Landscape:

            X, Y = _flip_axis(X, Y)

        X, Y = _bound(im.shape, X, Y)
        section = im[X[0]: X[1], Y[0]: Y[1]]

        return self._flip_short_dimension(section, im.shape)

    def _flip_short_dimension(self, section, im_shape):

        short_dim = [p == min(im_shape) for
                     p in im_shape].index(True)

        slicer = [i != short_dim and slice(None, None, None) or
                  slice(None, None, -1) for i in range(len(im_shape))]

        return section[slicer]

    def _set_current_grid_move(self, d1, d2):

        self._grid_corrections = np.array((d1, d2))

    def get_analysis(self, image_model):

        self.load_image(image_model.path)

        if self._im_loaded is False:
            return None

        if not image_model.grayscale_values:
            return None

        if not image_model.grayscale_target:
            image_model.grayscale_target = self.fixture['grayscaleTarget']
            if not image_model.grayscale_target:
                return None

        for plate in image_model.plates:

            if plate.index in self._grid_arrays:

                im = self.get_im_section(plate)
                grid_array = self._grid_arrays[plate.index]
                grid_array.doAnalysis(im, image_model)

                self.features[plate.index] = grid_array.features

        if self._analysis_model.focus_position:
            self._record_focus_colony_data()

        return self.features

    def _record_focus_colony_data(self):

        focus_plate = self._grid_arrays[self._analysis_model.focus_position[0]]

        self.watch_grid_size = focus_plate._grid_cell_size
        self.watch_source = focus_plate.watch_source
        self.watch_blob = focus_plate.watch_blob
        self.watch_results = focus_plate.watch_results