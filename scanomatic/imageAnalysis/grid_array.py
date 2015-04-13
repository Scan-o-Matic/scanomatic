#!/usr/bin/env python
"""Part of analysis work-flow that holds a grid arrays"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
import os
from matplotlib import pyplot as plt

#
# SCANNOMATIC LIBRARIES
#

import grid
from grid_cell import GridCell
import scanomatic.io.paths as paths
import scanomatic.io.logger as logger
import imageBasics
from scanomatic.models.analysis_model import IMAGE_ROTATIONS

#
# EXCEPTIONS


class InvalidGridException(Exception):
    pass


def _analyse_grid_cell(grid_cell, im, transpose_polynomial, features):

    grid_cell.source = _get_image_slice(im, grid_cell)

    if transpose_polynomial is not None:
        _set_image_transposition(grid_cell, transpose_polynomial)

    if not grid_cell.ready:
        grid_cell.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

    features[grid_cell.position] = grid_cell.get_analysis(remember_filter=True)


def _set_image_transposition(grid_cell, transpose_polynomial):

    grid_cell.source[...] = transpose_polynomial(grid_cell.source)


def _get_image_slice(im, grid_cell):

    xy1 = grid_cell.xy1
    xy2 = grid_cell.xy2

    return im[xy1[0]: xy2[0], xy1[1]: xy2[1]].copy()


def _create_grid_array_identifier(identifier):

    if isinstance(identifier, int):

        identifier = ("unknown", identifier)

    elif len(identifier) == 1:

        identifier = ["unknown", identifier[0]]

    else:

        identifier = [identifier[0], identifier[1]]

    return identifier


def make_grid_im(im, grid_corners, save_grid_name=None, x_values=None, y_values=None):

    grid_image = plt.figure()
    grid_plot = grid_image.add_subplot(111)
    grid_plot.imshow(im, cmap=plt.cm.gray)

    for row in xrange(grid_corners.shape[1]):

        grid_plot.plot(
            grid[1, row, :],
            grid[0, row, :],
            'r-')

    for col in xrange(grid_corners.shape[2]):

        grid_plot.plot(
            grid[1, :, col],
            grid[0, :, col],
            'r-')

    grid_plot.plot(grid_corners[1, 0, 0],
                   grid_corners[0, 0, 0],
                   'o', alpha=0.75, ms=10, mfc='none', mec='blue', mew=1)

    if x_values is not None and y_values is not None:

        grid_plot.plot(y_values, x_values, 'o', alpha=0.75,
                       ms=5, mfc='none', mec='red', mew=1)

    ax = grid_image.gca()
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(0, im.shape[0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if save_grid_name is None:

        grid_image.show()
        return grid_image

    else:

        grid_image.savefig(save_grid_name, pad_inches=0.01,
                           format='svg', bbox_inches='tight')

        grid_image.clf()
        plt.close(grid_image)
        del grid_image


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

    shuffle = pm_max_pos != im_max_pos
    return [int(shuffle), int(not shuffle)]

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

            if rotation is IMAGE_ROTATIONS.None:
                continue

            elif item in GridCellSizes._APPROXIMATE_GRID_CELL_SIZES:
                approximate_size = GridCellSizes._APPROXIMATE_GRID_CELL_SIZES[item]
                if rotation is IMAGE_ROTATIONS.Portrait:
                    approximate_size = approximate_size[reverse_slice]

            elif rotation is IMAGE_ROTATIONS.Portrait:
                # noinspection PyTypeChecker
                item = item[reverse_slice]

        if not approximate_size:
            GridCellSizes._LOGGER.warning("Unknown pinning format {0}".format(item))

        return approximate_size


class GridArray():

    def __init__(self, image_identifier, pinning, fixture, analysis_model):

        self._paths = paths.Paths()

        self.fixture = fixture
        self._identifier = _create_grid_array_identifier(image_identifier)
        self._analysis_model = analysis_model
        self._pinning_matrix = pinning

        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        self._guess_grid_cell_size = None
        self._grid_cell_size = None
        self._grid_cells = {}
        self._grid = None
        self._grid_cell_corners = None

        self.features = {}
        self._first_analysis = True

    @property
    def grid_cell_size(self):
        return self._grid_cell_size

    @property
    def index(self):
        return self._identifier[-1]

    def set_grid(self, im, save_name=None, grid_correction=None):

        self._init_grid_cells(_get_grid_to_im_axis_mapping(self._pinning_matrix, im))

        spacings = self._calculate_grid_and_get_spacings(im, grid_correction=grid_correction)

        if self._grid is None or np.isnan(spacings).any():

            error_file = os.path.join(
                os.path.dirname(self._analysis_model.first_pass_file),
                self._paths.experiment_grid_error_image.format(self._identifier[1]))

            np.save(error_file, im)
            save_name = error_file + ".png"
            make_grid_im(im, grid, save_grid_name=save_name.format(self.index))

            return False

        if not self._is_valid_grid_shape():

            raise InvalidGridException(
                "Grid shape {0} missmatch with pinning matrix {1}".format(self._grid.shape, self._pinning_matrix))

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)

        if save_name is not None:
            make_grid_im(im, self._grid_cell_corners, save_grid_name=save_name)

        self._set_grid_cell_corners()
        self._update_grid_cells()

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

        self._grid_cell_corners = np.zeros((2, self._grid.shape[1] + 1, self._grid.shape[2] + 1))

        # For both dimensions sets higher value boundaries
        self._grid_cell_corners[0, 1:, 1:] = self._grid[0] + self._grid_cell_size[0] / 2.0
        self._grid_cell_corners[1, 1:, 1:] = self._grid[1] + self._grid_cell_size[1] / 2.0
        # For all but the far right and bottom over-writes and sets lower values boundaries
        self._grid_cell_corners[0, :-1, :-1] = self._grid[0] - self._grid_cell_size[0] / 2.0
        self._grid_cell_corners[1, :-1, :-1] = self._grid[1] - self._grid_cell_size[1] / 2.0

    def _update_grid_cells(self):

        for grid_cell in self._grid_cells.values():

            grid_cell.set_grid_coordinates(self._grid_cell_corners)

    def _init_grid_cells(self, dimension_order=(0, 1)):

        self._pinning_matrix = (self._pinning_matrix[dimension_order[0]], self._pinning_matrix[dimension_order[1]])
        pinning_matrix = self._pinning_matrix

        self._guess_grid_cell_size = GridCellSizes.get(pinning_matrix)
        self._grid = None
        self._grid_cell_size = None
        self._grid_cells.clear()
        self.features.clear()
        polynomial_coeffs = get_calibration_polynomial_coeffs()

        for row in xrange(pinning_matrix[0]):

            for column in xrange(pinning_matrix[1]):

                if (not self._analysis_model.suppress_non_focal or
                        self._analysis_model.focus_position == (self._identifier[1], row, column)):

                    grid_cell = GridCell([self._identifier, [row, column]], polynomial_coeffs)
                    self._grid_cells[grid_cell.position] = grid_cell

    def analyse(self, im, image_model, save_grid_name=None):

        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        self._identifier[0] = image_model.index

        # noinspection PyBroadException
        try:
            transpose_polynomial = imageBasics.Image_Transpose(
                sourceValues=image_model.grayscale_values,
                targetValues=image_model.grayscale_targets)

        except Exception:

            transpose_polynomial = None

        if self._grid is None:
            if not self.set_grid(im):
                return None

        if save_grid_name:
            make_grid_im(im, self._grid_cell_corners, save_grid_name=save_grid_name)

        for grid_cell in self._grid_cells.values():
            _analyse_grid_cell(grid_cell, im, transpose_polynomial, self.features)

        self._set_focus_colony_results()

        return self.features

    def _set_focus_colony_results(self):

        if self._analysis_model.focus_position:

            grid_cell = self._grid_cells[tuple(self._analysis_model.focus_position[1:])]

            self.watch_blob = grid_cell.get_item('blob').filter_array.copy()

            self.watch_source = grid_cell.get_item('blob').grid_array.copy()
            self.watch_results = self.features[grid_cell.position]