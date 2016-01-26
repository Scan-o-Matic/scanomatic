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
from scanomatic.imageAnalysis.grayscale import getGrayscale
from scanomatic.models.factories.analysis_factories import AnalysisFeaturesFactory

#
# EXCEPTIONS


class InvalidGridException(Exception):
    pass


def _analyse_grid_cell(grid_cell, im, transpose_polynomial, image_index):

    """

    :type grid_cell: scanomatic.imageAnalysis.grid_cell.GridCell
    """
    grid_cell.source = _get_image_slice(im, grid_cell).astype(np.float64)
    grid_cell.image_index = image_index

    if transpose_polynomial is not None:
        _set_image_transposition(grid_cell, transpose_polynomial)

    if not grid_cell.ready:
        grid_cell.attach_analysis(
            blob=True, background=True, cell=True,
            run_detect=False)

    grid_cell.analyse(remember_filter=True)


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


def make_grid_im(im, grid_corners, save_grid_name=None, x_values=None, y_values=None, origin=None):

    grid_image = plt.figure()
    grid_plot = grid_image.add_subplot(111)
    grid_plot.imshow(im.T, cmap=plt.cm.gray)
    x = 0
    y = 1

    if grid_corners is not None:

        for row in range(grid_corners.shape[2]):

            grid_plot.plot(
                grid_corners[x, :, row, :].mean(axis=0),
                grid_corners[y, :, row, :].mean(axis=0),
                'r-')

        for col in range(grid_corners.shape[3]):

            grid_plot.plot(
                grid_corners[x, :, :, col].mean(axis=0),
                grid_corners[y, :, :, col].mean(axis=0),
                'r-')

        if origin:

            pos = np.mean((origin.xy1, origin.xy2), axis=0)
            grid_plot.plot(pos[0], pos[1], 'o', alpha=0.75, ms=10, mfc='none', mec='blue', mew=1)

    if x_values is not None and y_values is not None:

        grid_plot.plot(x_values, y_values, 'o', alpha=0.75,
                       ms=5, mfc='none', mec='red', mew=1)

    ax = grid_image.gca()
    ax.set_xlim(0, im.shape[x])
    ax.set_ylim(im.shape[y], 0)
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


class GridArray():

    _LOGGER = logger.Logger("Grid Array")

    def __init__(self, image_identifier, pinning, analysis_model):

        self._paths = paths.Paths()

        self._identifier = _create_grid_array_identifier(image_identifier)
        self._analysis_model = analysis_model
        self._pinning_matrix = pinning

        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        self._guess_grid_cell_size = None
        self._grid_cell_size = None
        self._grid_cells = {}
        """:type:dict[tuple|scanomatic.imageAnalysis.grid_cell.GridCell]"""
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
        self._features.index[0] = value

    def set_grid(self, im, save_name=None, grid_correction=None):

        self._init_grid_cells(_get_grid_to_im_axis_mapping(self._pinning_matrix, im))

        spacings = self._calculate_grid_and_get_spacings(im, grid_correction=grid_correction)

        if self._grid is None or np.isnan(spacings).any():

            error_file = os.path.join(
                self._analysis_model.output_directory,
                self._paths.experiment_grid_error_image.format(self.index))

            np.save(error_file, im)
            save_name = error_file + ".svg"
            make_grid_im(im, self._grid_cell_corners, save_grid_name=save_name.format(self.index))

            return False

        if not self._is_valid_grid_shape():

            raise InvalidGridException(
                "Grid shape {0} missmatch with pinning matrix {1}".format(self._grid.shape, self._pinning_matrix))

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)
        self._set_grid_cell_corners()
        self._update_grid_cells()

        if save_name is not None:
            save_name += "{0}.svg".format(self.index + 1)
            make_grid_im(im, self._grid_cell_corners, save_grid_name=save_name, origin=self._grid_cells[(0, 0)])

            np.save(os.path.join(os.path.dirname(save_name),
                                 self._paths.grid_pattern.format(self.index + 1)), self._grid)

            np.save(os.path.join(os.path.dirname(save_name),
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

        for row in xrange(pinning_matrix[0]):

            for column in xrange(pinning_matrix[1]):

                if (not self._analysis_model.suppress_non_focal or
                        self._analysis_model.focus_position == (self.index, row, column)):

                    grid_cell = GridCell([self._identifier, (row, column)], polynomial_coeffs)
                    self._features.data.add(grid_cell.features)
                    self._grid_cells[grid_cell.position] = grid_cell

    def clear_features(self):
        for grid_cell in self._grid_cells.itervalues():
            grid_cell.clear_features()

    def analyse(self, im, image_model, save_grid_name=None):

        """

        :type image_model: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        index = image_model.image.index
        self.image_index = index
        self._LOGGER.info("Processing {0}, index {1}".format(self._identifier, index))

        # noinspection PyBroadException
        try:
            transpose_polynomial = imageBasics.Image_Transpose(
                sourceValues=image_model.fixture.grayscale.values,
                targetValues=getGrayscale(image_model.fixture.grayscale.name)['targets'])

        except Exception:

            transpose_polynomial = None

        if self._grid is None:
            if not self.set_grid(im):
                self.clear_features()
                return

        if save_grid_name:
            make_grid_im(im, self._grid_cell_corners, save_grid_name=save_grid_name)

        for grid_cell in self._grid_cells.values():
            _analyse_grid_cell(grid_cell, im, transpose_polynomial, index)

        self._set_focus_colony_results()


    def _set_focus_colony_results(self):

        if self._analysis_model.focus_position:

            grid_cell = self._grid_cells[tuple(self._analysis_model.focus_position[1:])]

            self.watch_blob = grid_cell.get_item('blob').filter_array.copy()

            self.watch_source = grid_cell.get_item('blob').grid_array.copy()
            self.watch_results = self.features[grid_cell.position]