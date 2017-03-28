import os
import numpy as np
from time import sleep
from threading import Thread
from subprocess import call

#
# SCANNOMATIC LIBRARIES
#

import grid_array
from image_grayscale import is_valid_grayscale
from grayscale import getGrayscale
from image_basics import load_image_to_numpy
from scanomatic.io.logger import Logger
from scanomatic.models.analysis_model import IMAGE_ROTATIONS
from scanomatic.models.factories.analysis_factories import AnalysisFeaturesFactory
from scanomatic.io.paths import Paths

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
        data=tuple(grid_arrays[i].features if i in grid_arrays else None for i in range(size)),
        index=0)

    return features


class ProjectImage(object):

    def __init__(self, analysis_model, first_pass_results):
        """

        :param analysis_model: The model
         :type analysis_model : scanomatic.models.analysis_model.AnalysisModel
        :param first_pass_results: The results of project compilation
         :type first_pass_results: scanomatic.io.first_pass_results.CompilationResults
        """

        self._analysis_model = analysis_model
        self._first_pass_results = first_pass_results

        self._logger = Logger("Analysis Image")

        self._im_loaded = False
        self.im = None
        self._im_path_as_requested = None

        self._grid_arrays = self._new_grid_arrays
        self._plate_image_inclusion = self.image_inclusions

        """:type : dict[int|scanomatic.image_analysis.grid_array.GridArray]"""
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

    @property
    def image_inclusions(self):

        all_images = set(self._first_pass_results.keys())
        highest_index_plus_one = max(all_images) + 1

        if self._analysis_model.plate_image_inclusion is None:

            self._logger.info("No plate specific image inclusion assumes all plate included for images {0}".format(
                all_images))

            return {k: all_images for k in self._grid_arrays}

        else:

            ret = {}
            platewise_len = len(self._analysis_model.plate_image_inclusion)

            for i in range(max(self._grid_arrays.keys(), platewise_len)):

                if i not in self._grid_arrays or i < 0 or i >= platewise_len:

                    if platewise_len > i > 0:
                        if self._analysis_model.plate_image_inclusion[i] is not None:
                            self._logger.warning(
                                "There's a image selection for plate index {0}, but this plate does not exist".format(
                                    i))
                    else:
                        self._logger.warning(
                            "Plate index {0} has no instructions for inclusion of images, assuming all included".format(
                                i))
                        ret[i] = all_images

                else:
                    ret[i] = set()
                    instruction = self._analysis_model.plate_image_inclusion[i]
                    instruction = [[val.strip() for val in part.strip().split("-")] for part in instruction.split(",")]

                    if not all(len(part) == 2 for part in instruction):
                        self._logger.error("Malformed plate inclusion settings: '{0}'".format(
                            self._analysis_model.plate_image_inclusion[i]) + " Plate excluded from analysis")

                        continue

                    try:
                        instruction = [(int(start) if start else 0, int(end) + 1 if end else highest_index_plus_one)
                                       for start, end in instruction]
                    except ValueError:
                        self._logger.error("Plate inclusion setting contains non-ints {0}".format(instruction) +
                                           " Plate excluded from analysis")
                        continue

                    for start, end in instruction:
                        ret[i].update(range(start, end))

            return ret

    def _plate_is_analysed(self, index):

        return not self._analysis_model.suppress_non_focal or index == self._analysis_model.focus_position[0]

    def _get_index_for_gridding(self):

        if self._analysis_model.grid_images:
            pos = max(self._analysis_model.grid_images)
            if pos >= len(self._first_pass_results):
                pos = self._first_pass_results.last_index
        else:

            pos = self._first_pass_results.last_index

        return pos

    def set_grid(self):
        """Sets grids if same index for everyone"""

        if self._analysis_model.plate_image_inclusion is not None:
            # This should be true because it is alright, gridding will be fixed during analysis instead
            return True

        image_model = self._first_pass_results[self._get_index_for_gridding()]

        return self.set_grid_plates(self._grid_arrays.keys(), image_model)

    def set_grid_plates(self, plate_indices, image_model):

        if image_model is None:
            self._logger.critical("No image model to grid on")
            return False

        self.load_image(image_model.image.path)

        if self._im_loaded:

            threads = set()

            self._logger.info("Setting grids for plates {0} using image index {1}".format(
                plate_indices, image_model.image.index))

            for index in plate_indices:

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

                if self._analysis_model.grid_model.gridding_offsets is not None and \
                        index < len(self._analysis_model.grid_model.gridding_offsets) \
                        and self._analysis_model.grid_model.gridding_offsets[index]:

                    reference_folder = self._analysis_model.grid_model.reference_grid_folder
                    if reference_folder:
                        reference_folder = os.path.join(os.path.dirname(self._analysis_model.output_directory),
                                                        reference_folder)
                    else:
                        reference_folder = self._analysis_model.output_directory

                    if not self._grid_arrays[index].set_grid(
                            im, analysis_directory=self._analysis_model.output_directory,
                            offset=self._analysis_model.grid_model.gridding_offsets[index],
                            grid=os.path.join(reference_folder, Paths().grid_pattern.format(index + 1))):

                        self._logger.error("Could not use previous gridding with offset")

                else:

                    t = Thread(target=self._grid_arrays[index].detect_grid,
                               args=(im,), kwargs=dict(analysis_directory=self._analysis_model.output_directory))
                    t.start()
                    threads.add(t)

            while threads:
                threads = set(t for t in threads if t.is_alive())
                sleep(0.01)

            self._logger.info("Producing grid images for plates {0} based on image {1} and compilation '{2}'".format(
                plate_indices, image_model.image.path, self._analysis_model.compilation))

            call(["python",
                  "-c",
                  "from scanomatic.util.analysis import produce_grid_images;"
                  " produce_grid_images('{0}', plates={1}, compilation='{2}')".format(
                      self._analysis_model.output_directory, plate_indices, self._analysis_model.compilation)])
        else :

            self._logger.warning("No gridding done for plates {0} because image not loaded.".format(plate_indices))

        return True

    def load_image(self, path):

        if path == self._im_path_as_requested:
            self._logger.info("Image was already loaded")
            return

        try:

            self.im = load_image_to_numpy(path, IMAGE_ROTATIONS.Portrait, dtype=np.uint8)
            self._im_loaded = True

        except (TypeError, IOError):

            alt_path = os.path.join(os.path.dirname(self._analysis_model.compilation),
                                    os.path.basename(path))

            self._logger.warning("Failed to load image at '{0}', trying '{1}'.".format(path, alt_path))
            try:

                self.im = load_image_to_numpy(alt_path, IMAGE_ROTATIONS.Portrait, dtype=np.uint8)
                self._im_loaded = True

            except (TypeError, IOError):

                self._im_loaded = False

        if self._im_loaded:
            self._logger.info("Image loaded")
            self._im_path_as_requested = path
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
        self._logger.info("Image loaded")
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
        # threads = set()
        for plate in image_model.fixture.plates:

            if plate.index in self._grid_arrays:

                if image_model.image.index not in self._plate_image_inclusion[plate.index]:
                    self._logger.info("Skipping image {0} on plate {1} due to inclusion settings".format(
                        image_model.image.index, plate.index
                    ))
                    continue

                if not self._grid_arrays[plate.index].has_grid:
                    self.set_grid_plates([plate.index], image_model)

                grid_arrays_processed.add(plate.index)
                im = self.get_im_section(plate)
                grid_arr = self._grid_arrays[plate.index]
                """:type: scanomatic.image_analysis.grid_array.GridArray"""
                grid_arr.analyse(im, image_model)
                """
                t = Thread(target=grid_arr.analyse, args=(im, image_model))
                t.start()
                threads.add(t)

        while threads:
            threads = set(t for t in threads if t.is_alive())
            sleep(0.01)
        """
        for index, grid_array in self._grid_arrays.iteritems():
            if index not in grid_arrays_processed:
                grid_array.clear_features()

        self._logger.info("Image {0} processed".format(image_model.image.index))
