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
from imageGrayscale import is_valid_grayscale
from grayscale import getGrayscale
from scanomatic.io.logger import Logger
from scanomatic.models.analysis_model import IMAGE_ROTATIONS
from scanomatic.models.factories.analysis_factories import AnalysisFeaturesFactory


#
# CLASS Project_Image
#


# noinspection PyTypeChecker
def _get_init_features(grid_arrays):

    """

    :type grid_arrays: dict[int|scanomatic.imageAnalysis.grid_array.GridArray]
    :rtype : list[None|dict[str|object]]
    """

    def length_needed(keys):

        return max(keys) + 1

    size = length_needed(grid_arrays.keys()) if grid_arrays else 0

    features = AnalysisFeaturesFactory.create(
        shape=(size,),
        data=tuple(grid_arrays[i].features if grid_arrays[i] else None for i in range(size)),
        index=0)

    return features



class ProjectImage(object):

    def __init__(self, analysis_model, scanning_meta_data):

        self._analysis_model = analysis_model
        self._scanning_meta_data = scanning_meta_data
        self._logger = Logger("Analysis Image")

        self._im_loaded = False
        self.im = None

        self._grid_arrays = self._new_grid_arrays
        """:type : dict[int|scanomatic.imageAnalysis.grid_array.GridArray]"""
        self.features = _get_init_features(self._grid_arrays)

    @property
    def active_plates(self):
        return len(self._grid_arrays)

    def __getitem__(self, key):

        return self._grid_arrays[key]

    @property
    def _new_grid_arrays(self):

        """

        :rtype : dict[int|scanomatic.imageAnalysis.grid_array.GridArray]
        """
        grid_arrays = {}

        for index, pinning in enumerate(self._analysis_model.pinning_matrices):

            if pinning and self._plate_is_analysed(index):

                grid_arrays[index] = grid_array.GridArray(index, pinning, self._analysis_model)

            else:

                if pinning:

                    self._logger.info("Skipping plate {0} because suppressing non-focal positions".format(index))

                else:

                    self._logger.info("Plate {0} not analysed because lacks pinning".format(index))

        return grid_arrays

    def _plate_is_analysed(self, index):

        return not self._analysis_model.suppress_non_focal or index == self._analysis_model.focus_position[0]

    def set_grid(self, image_model, save_name=None):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        if image_model is None:
            self._logger.critical("No image model to grid on")
            return False

        if save_name is None:
            save_name = os.sep.join((self._analysis_model.output_directory, "grid___origin_plate_"))

        self.load_image(image_model.image.path)

        if self._im_loaded:

            for index in self._grid_arrays:

                plate_models = [plate_model for plate_model in image_model.fixture.plates if plate_model.index == index]
                if plate_models:
                    plate_model = plate_models[0]
                else:
                    self._logger.error("Expected to find a plate model with index {0}, but only have {1}".format(
                        index, [plate_model.index for plate_model in image_model.fixture.plates]))
                    continue

                im = self.get_im_section(plate_model)

                if im is None:
                    self._logger.error("Plate model {0} could not be used to slice image".format(plate_model))
                    continue
                if not self._grid_arrays[index].set_grid(
                        im, save_name=save_name,
                        grid_correction=self._analysis_model.grid_model.gridding_offsets[index]
                        if self._analysis_model.grid_model.gridding_offsets is not None and
                        index < len(self._analysis_model.grid_model.gridding_offsets) else None):

                    self._logger.warning("Failed to grid plate {0}".format(plate_model))

        return True

    def load_image(self, path):

        try:

            self.im = plt_img.imread(path)
            self._im_loaded = True

        except (TypeError, IOError):

            alt_path = os.path.join(os.path.dirname(self._analysis_model.first_pass_file),
                                    os.path.basename(path))

            self._logger.warning("Failed to load image at '{0}', trying '{1}'.".format(
                path, alt_path
            ))
            try:

                self.im = plt_img.imread(alt_path)
                self._im_loaded = True

            except (TypeError, IOError):

                self._im_loaded = False

        if self._im_loaded:
            self._logger.info("Image loaded")
        else:
            self._logger.error("Failed to load image")

        self.validate_rotation()
        self._convert_to_grayscale()

    def _convert_to_grayscale(self):
        if self.im.ndim == 3:
            self.im = np.dot(self.im[..., :3], [0.299, 0.587, 0.144])

    def validate_rotation(self):

        pass

    @property
    def orientation(self):
        """The currently loaded image's rotation considered as first dimension of image array being image rows
        :return:
        """
        if not self._im_loaded:
            return IMAGE_ROTATIONS.Unknown
        elif self.im.shape[0] > self.im.shape[1]:
            return IMAGE_ROTATIONS.Portrait
        else:
            return IMAGE_ROTATIONS.Landscape

    def get_im_section(self, plate_model, im=None):

        def _flip_axis(a, b):

            return b, a

        def _bound(bounds, a, b):

            def bounds_check(bound, val):

                if 0 <= val < bound:
                    return val
                elif val < 0:
                    return 0
                else:
                    return bound - 1

            return ((bounds_check(bounds[0], a[0]),
                     bounds_check(bounds[0], a[1])),
                    (bounds_check(bounds[1], b[0]),
                     bounds_check(bounds[1], b[1])))

        if not im:
            if self._im_loaded:
                im = self.im
            else:
                return

        x = sorted((plate_model.x1, plate_model.x2))
        y = sorted((plate_model.y1, plate_model.y2))

        if self.orientation == IMAGE_ROTATIONS.Landscape:
            x, y = _flip_axis(x, y)

        y, x = _bound(im.shape, y, x)

        # In images, the first dimension is typically the y-axis
        section = im[y[0]: y[1], x[0]: x[1]]

        return self._flip_short_dimension(section, im.shape)

    @staticmethod
    def _flip_short_dimension(section, im_shape):

        short_dim = [p == min(im_shape) for
                     p in im_shape].index(True)

        def get_slicer(idx):
            if idx == short_dim:
                # noinspection PyTypeChecker
                return slice(None, None, -1)
            else:
                # noinspection PyTypeChecker
                return slice(None)

        slicer = []
        for i in range(len(im_shape)):
            slicer.append(get_slicer(i))

        return section[slicer]

    def _set_current_grid_move(self, d1, d2):

        self._grid_corrections = np.array((d1, d2))

    def clear_features(self):
        for grid_array in self._grid_arrays.itervalues():
            grid_array.clear_features()

    def analyse(self, image_model):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        self.load_image(image_model.image.path)

        if self._im_loaded is False:
            self.clear_features()
            return

        if not image_model.fixture.grayscale.values or not is_valid_grayscale(
                getGrayscale(image_model.fixture.grayscale.name)['targets'],
                image_model.fixture.grayscale.values):

            self._logger.warning("Not a valid grayscale")
            self.clear_features()
            return

        self.features.index = image_model.image.index
        grid_arrays_processed = set()
        for plate in image_model.fixture.plates:

            if plate.index in self._grid_arrays:
                grid_arrays_processed.add(plate.index)
                im = self.get_im_section(plate)
                grid_arr = self._grid_arrays[plate.index]
                """:type: scanomatic.imageAnalysis.grid_array.GridArray"""
                grid_arr.analyse(im, image_model)

        for index, grid_array in self._grid_arrays.iteritems():
            if index not in grid_arrays_processed:
                grid_array.clear_features()

        if self._analysis_model.focus_position:
            self._record_focus_colony_data()

    def _record_focus_colony_data(self):

        focus_plate = self._grid_arrays[self._analysis_model.focus_position[0]]

        self.watch_grid_size = focus_plate.grid_cell_size
        self.watch_source = focus_plate.watch_source
        self.watch_blob = focus_plate.watch_blob
        self.watch_results = focus_plate.watch_results