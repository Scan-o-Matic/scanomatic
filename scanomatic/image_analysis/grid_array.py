#
# DEPENDENCIES
#

import numpy as np
import os


#
# SCANNOMATIC LIBRARIES
#

import grid
from grid_cell import GridCell
import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
import image_basics
from scanomatic.models.analysis_model import IMAGE_ROTATIONS
from scanomatic.image_analysis.grayscale import getGrayscale
from scanomatic.models.factories.analysis_factories import AnalysisFeaturesFactory
#
# EXCEPTIONS


class InvalidGridException(Exception):
    pass


def _analyse_grid_cell(grid_cell, im, transpose_polynomial, image_index, semaphore=None, analysis_job_model=None):

    """

    :type grid_cell: scanomatic.imageAnalysis.grid_cell.GridCell
    """
    save_extra_data = grid_cell.save_extra_data

    grid_cell.source = _get_image_slice(im, grid_cell).astype(np.float64)
    grid_cell.image_index = image_index

    if save_extra_data:
        grid_cell.save_data_image(suffix=".raw",
                                  base_path=analysis_job_model.output_directory if analysis_job_model else None)


    if transpose_polynomial is not None:
        _set_image_transposition(grid_cell, transpose_polynomial)

    if save_extra_data:
        grid_cell.save_data_image(suffix=".calibrated",
                                  base_path=analysis_job_model.output_directory if analysis_job_model else None)

    if not grid_cell.ready:
        grid_cell.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

    # TODO: Deterimine if it is best to remember history or not!
    grid_cell.analyse(remember_filter=False)

    if save_extra_data:
        grid_cell.save_data_detections(base_path=analysis_job_model.output_directory if analysis_job_model else None)

    if semaphore is not None:
        semaphore.release()


def _set_image_transposition(grid_cell, transpose_polynomial):

    grid_cell.source[...] = transpose_polynomial(grid_cell.source)


def _get_image_slice(im, grid_cell):

    """

    :type grid_cell: scanomatic.imageAnalysis.grid_cell.GridCell
    """
    xy1 = grid_cell.xy1
    xy2 = grid_cell.xy2

    return im[xy1[0]: xy2[0], xy1[1]: xy2[1]].copy()


def _create_grid_array_identifier(identifier):

    no_image_reference = "unknown image"
    if isinstance(identifier, int):

        identifier = [no_image_reference, identifier]

    elif len(identifier) == 1:

        identifier = [no_image_reference, identifier[0]]

    else:

        identifier = [identifier[0], identifier[1]]

    return identifier


def get_calibration_polynomial_coeffs():

    try:

        fs = open(paths.Paths().analysis_polynomial, 'r')

    except IOError:

        return None

    polynomial_coeffs = []

    for l in fs:

        l_data = eval(l.strip("\n"))

        if isinstance(l_data, list):

            polynomial_coeffs = l_data[-1]
            break

    fs.close()

    if not polynomial_coeffs:

        return None

    return polynomial_coeffs


def _get_grid_to_im_axis_mapping(pm, im):

    pm_max_pos = int(max(pm) == pm[1])
    im_max_pos = int(max(im.shape) == im.shape[1])

    if pm_max_pos == im_max_pos:
        return (0, 1)
    else:
        return (1, 0)

#
# CLASS: Grid_Array
#


class GridCellSizes(object):

    _LOGGER = logger.Logger("Grid Cell Sizes")

    _APPROXIMATE_GRID_CELL_SIZES = {
        (8, 12): (212, 212),
        (16, 24): (106, 106),
        (32, 48): (53.64928854, 52.69155633),
        (64, 96): (40.23696640, 39.5186672475),
    }

    @staticmethod
    def get(item):
        """

        :type item: tuple
        """
        if not isinstance(item, tuple):
            GridCellSizes._LOGGER.error("Grid formats can only be tuples {0}".format(type(item)))
            return None

        approximate_size = None
        # noinspection PyTypeChecker
        reverse_slice = slice(None, None, -1)

        for rotation in IMAGE_ROTATIONS:

            if rotation is IMAGE_ROTATIONS.Unknown:
                continue

            elif item in GridCellSizes._APPROXIMATE_GRID_CELL_SIZES:
                approximate_size = GridCellSizes._APPROXIMATE_GRID_CELL_SIZES[item]
                if rotation is IMAGE_ROTATIONS.Portrait:
                    approximate_size = approximate_size[reverse_slice]
                break
            else:
                item = item[reverse_slice]

        if not approximate_size:
            GridCellSizes._LOGGER.warning("Unknown pinning format {0}".format(item))

        return approximate_size


class GridArray(object):

    _LOGGER = logger.Logger("Grid Array")

    def __init__(self, image_identifier, pinning, analysis_model):

        self._paths = paths.Paths()

        self._identifier = _create_grid_array_identifier(image_identifier)
        self._analysis_model = analysis_model
        self._pinning_matrix = pinning

        self._guess_grid_cell_size = None
        self._grid_cell_size = None
        self._grid_cells = {}
        """:type:dict[tuple|scanomatic.image_analysis.grid_cell.GridCell]"""
        self._grid = None
        self._grid_cell_corners = None

        self._features = AnalysisFeaturesFactory.create(index=self._identifier[-1], shape=tuple(pinning), data=set())
        self._first_analysis = True

    @property
    def features(self):
        return self._features

    @property
    def grid_cell_size(self):
        return self._grid_cell_size

    @property
    def index(self):
        return self._identifier[-1]

    @property
    def image_index(self):
        return self._identifier[0]

    @image_index.setter
    def image_index(self, value):
        self._identifier[0] = value

    def set_grid(self, im, analysis_directory=None, offset=None, grid=None):

        self._LOGGER.info("Setting manual re-gridding for plate {0} using offset {1} on reference grid {2}".format(
            self.index + 1, offset, grid))

        if not offset:
            return self.detect_grid(im, analysis_directory=analysis_directory, grid_correction=offset)

        try:
            grid = np.load(grid)
        except IOError:
            self._LOGGER.error("No grid file named '{0}'".format(grid))
            self._LOGGER.info("Invoking grid detection instead")
            return self.detect_grid(im, analysis_directory=analysis_directory, grid_correction=offset)

        self._init_grid_cells(_get_grid_to_im_axis_mapping(self._pinning_matrix, im))

        spacings = ((grid[0, 1:] - grid[0, :-1]).ravel().mean(), (grid[1, :, 1:] - grid[1, :, :-1]).ravel().mean())

        if offset and not all(o == 0 for o in offset):

            # The direction of the first axis is flipped to make offsetting more logical from user perspective
            # this inversion must be matched by equal inversion in detect_grid

            o = offset[0] * -1
            delta = spacings[0]
            if o > 0:

                grid[0, :-o] = grid[0, o:]
                for idx in range(-o, 0):
                    grid[0, idx] = grid[0, idx - 1] + delta

            elif o < 0:

                grid[0, -o:] = grid[0, :o]
                for idx in range(-o)[::-1]:
                    grid[0, idx] = grid[0, idx + 1] - delta

            o = offset[1]
            delta = spacings[1]
            if o > 0:

                grid[1, :, :-o] = grid[1, :, o:]
                for idx in range(-o, 0):
                    grid[1, :, idx] = grid[1, :, idx - 1] + delta

            elif o < 0:

                grid[1, :, -o:] = grid[1, :, :o]
                for idx in range(-o)[::-1]:
                    grid[1, :, idx] = grid[1, :, idx + 1] - delta

        self._grid = grid

        if not self._is_valid_grid_shape():

            raise InvalidGridException(
                "Grid shape {0} missmatch with pinning matrix {1}".format(self._grid.shape, self._pinning_matrix))

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)
        self._set_grid_cell_corners()
        self._update_grid_cells()

        if analysis_directory is not None:

            np.save(os.path.join(analysis_directory, self._paths.grid_pattern.format(self.index + 1)), self._grid)

            np.save(os.path.join(analysis_directory,
                                 self._paths.grid_size_pattern.format(self.index + 1)), self._grid_cell_size)
        return True

    def detect_grid(self, im, analysis_directory=None, grid_correction=None):

        self._LOGGER.info("Detecting grid on plate {0} using grid correction {1}".format(
            self.index + 1, grid_correction))

        # The direction of the first axis is flipped to make offsetting more logical from user perspective
        # this inversion must be matched by equal inversion in set_grid

        if grid_correction:
            grid_correction = list(grid_correction)
            grid_correction[0] *= -1

        self._init_grid_cells(_get_grid_to_im_axis_mapping(self._pinning_matrix, im))

        spacings = self._calculate_grid_and_get_spacings(im, grid_correction=grid_correction)

        if self._grid is None or np.isnan(spacings).any():

            error_file = os.path.join(
                self._analysis_model.output_directory,
                self._paths.experiment_grid_error_image.format(self.index))

            np.save(error_file, im)
            self._LOGGER.warning("Failed to detect grid on plate {0}".format(self.index))

            return False

        if not self._is_valid_grid_shape():

            raise InvalidGridException(
                "Grid shape {0} missmatch with pinning matrix {1}".format(self._grid.shape, self._pinning_matrix))

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)
        self._set_grid_cell_corners()
        self._update_grid_cells()

        if analysis_directory is not None:

            np.save(os.path.join(analysis_directory, self._paths.grid_pattern.format(self.index + 1)), self._grid)

            np.save(os.path.join(analysis_directory,
                                 self._paths.grid_size_pattern.format(self.index + 1)), self._grid_cell_size)

        return True

    def _calculate_grid_and_get_spacings(self, im, grid_correction=None):

        validate_parameters = False
        expected_spacings = self._guess_grid_cell_size
        expected_center = tuple([s / 2.0 for s in im.shape])

        draft_grid, _, _, _, spacings, adjusted_values = grid.get_grid(
            im,
            expected_spacing=expected_spacings,
            expected_center=expected_center,
            validate_parameters=validate_parameters,
            grid_shape=self._pinning_matrix,
            grid_correction=grid_correction)

        dx, dy = spacings

        self._grid, _ = grid.get_validated_grid(
            im, draft_grid, dy, dx, adjusted_values)

        return spacings

    def _is_valid_grid_shape(self):

        return all(g == i for g, i in zip(self._grid.shape[1:], self._pinning_matrix))

    def _set_grid_cell_corners(self):

        self._grid_cell_corners = np.zeros((2, 2, self._grid.shape[1], self._grid.shape[2]))

        # For all sets lower values boundaries
        self._grid_cell_corners[0, 0, :, :] = self._grid[0] - self._grid_cell_size[0] / 2.0
        self._grid_cell_corners[1, 0, :, :] = self._grid[1] - self._grid_cell_size[1] / 2.0

        # For both dimensions sets higher value boundaries
        self._grid_cell_corners[0, 1, :, :] = self._grid[0] + self._grid_cell_size[0] / 2.0
        self._grid_cell_corners[1, 1, :, :] = self._grid[1] + self._grid_cell_size[1] / 2.0

    def _update_grid_cells(self):

        for grid_cell in self._grid_cells.itervalues():

            grid_cell.set_grid_coordinates(self._grid_cell_corners)

    def _init_grid_cells(self, dimension_order=(0, 1)):

        self._pinning_matrix = (self._pinning_matrix[dimension_order[0]], self._pinning_matrix[dimension_order[1]])
        pinning_matrix = self._pinning_matrix

        self._guess_grid_cell_size = GridCellSizes.get(pinning_matrix)
        self._grid = None
        self._grid_cell_size = None
        self._grid_cells.clear()
        self._features.data.clear()

        polynomial_coeffs = get_calibration_polynomial_coeffs()
        focus_position = self._analysis_model.focus_position

        for row in xrange(pinning_matrix[0]):

            for column in xrange(pinning_matrix[1]):
                cur_position = (self.index, row, column)
                if not self._analysis_model.suppress_non_focal or focus_position == cur_position:

                    is_focus = focus_position == cur_position if focus_position else False
                    grid_cell = GridCell([self._identifier, (row, column)], polynomial_coeffs, save_extra_data=is_focus)
                    self._features.data.add(grid_cell.features)
                    self._grid_cells[grid_cell.position] = grid_cell

    def clear_features(self):
        for grid_cell in self._grid_cells.itervalues():
            grid_cell.clear_features()

    def analyse(self, im, image_model):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """

        index = image_model.image.index
        self.image_index = index
        self._LOGGER.info("Processing {0}, index {1}".format(self._identifier, index))

        # noinspection PyBroadException
        try:
            transpose_polynomial = image_basics.Image_Transpose(
                sourceValues=image_model.fixture.grayscale.values,
                targetValues=getGrayscale(image_model.fixture.grayscale.name)['targets'])

        except Exception:

            transpose_polynomial = None

        if self._grid is None:
            if not self.detect_grid(im):
                self.clear_features()
                return

        m = self._analysis_model

        for grid_cell in self._grid_cells.itervalues():

            if grid_cell.save_extra_data:
                self._LOGGER.info("Starting analysis of extra monitored position {0}".format(grid_cell.position))
            _analyse_grid_cell(grid_cell, im, transpose_polynomial, index, None, m)

        self._LOGGER.info("Plate {0} completed".format(self._identifier))
