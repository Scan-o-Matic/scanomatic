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
from grid_cell import Grid_Cell
import scanomatic.io.paths as paths
import imageBasics

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


class Grid_Array():

    _APPROXIMATE_GRID_CELL_SIZES = {
        (8, 12): (212, 212),
        (16, 24): (106, 106),
        (32, 48): (53.64928854, 52.69155633),
        (64, 96): (40.23696640, 39.5186672475),
        None: None
    }

    def __init__(self, image_identifier, pinning, fixture, analysis_model):

        self._paths = paths.Paths()

        self.fixture = fixture
        self._identifier = _create_grid_array_identifier(image_identifier)
        self._analysis_model = analysis_model
        self._pinning_matrix = pinning

        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        self._polynomial_coeffs = self.get_calibration_polynomial_coeffs()

        self._guess_grid_cell_size = None
        self._grid_cell_size = None
        self._grid_cells = []
        self._grid = None
        self._grid_cell_corners = None

        self.features = []
        self._first_analysis = True

    @property
    def __index__(self):
        return self._identifier[-1]

    def set_grid(self, im, save_name=None, grid_correction=None):

        self._init_pinning_matrix()

        if self._im_dim_order is None:

            self._im_dim_order = self._get_grid_to_im_axis_mapping(
                self._pinning_matrix, im)

        grid_shape = (self._pinning_matrix[int(self._im_dim_order[0])],
                      self._pinning_matrix[int(self._im_dim_order[1])])

        #If too little data, use very rough guesses
        if True:
            validate_parameters = False
            expected_spacings = self._guess_grid_cell_size
            expected_center = tuple([s / 2.0 for s in im.shape])


        grd, X, Y, center, spacings, adjusted_values = grid.get_grid(
            im,
            expected_spacing=expected_spacings,
            expected_center=expected_center,
            validate_parameters=validate_parameters,
            grid_shape=grid_shape,
            grid_correction=grid_correction)

        dx, dy = spacings
        self._grid, adjusted_values = grid.get_validated_grid(
            im, grd, dy, dx, adjusted_values)

        #self.logger.info("Expecting center {0} and Spacings {1}".format(
        #    expected_center, expected_spacings))

        if self._grid is None or np.isnan(spacings).any():
            #self.logger.error(
            #    "Could not produce a grid for im-shape {0}".format(im.shape))

            error_file = os.path.join(
                os.sep,
                self._parent().get_file_base_dir(),
                self._paths.experiment_grid_error_image.format(
                    self._identifier[1]))

            if not os.path.isfile(error_file):

                np.save(error_file, im)

                #self.logger.critical('Saved image slice to {0}'.format(
                #    error_file))

            else:

                #self.logger.critical("Won't save failed gridding {0}".format(
                #    error_file) + ", file allready exists")
                pass

            if save_name is not None:
                self.make_grid_im(im, grid, save_grid_name=save_name.format(self.index))

            return False

        if (self._grid.shape[1] != self._pinning_matrix[self._im_dim_order[0]]
                or self._grid.shape[2] !=
                self._pinning_matrix[self._im_dim_order[1]]):

            raise InvalidGridException(
                "Grid shape {0} missmatch with pinning matrix {1}".format(
                self._grid.shape,
                (self._pinning_matrix[self._im_dim_order[0]],
                self._pinning_matrix[self._im_dim_order[1]])))

        #self.logger.info("Got center {0} and Spacings {1}".format(
        #    center, spacings))

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)

        if save_name is not None:
            make_grid_im(im, self._grid_cell_corners, save_grid_name=save_name)

        self._set_grid_cell_corners()
        self._update_grid_cells()

        return True

    def _set_grid_cell_corners(self):

        self._grid_cell_corners = np.zeros((2, self._grid.shape[1] + 1, self._grid.shape[2] + 1))

        # For both dimensions sets higher value boundaries
        self._grid_cell_corners[0, 1:, 1:] =  self._grid[0] + self._grid_cell_size[0] / 2.0
        self._grid_cell_corners[1, 1:, 1:] = self._grid[1] + self._grid_cell_size[1] / 2.0
        # For all but the far right and bottom over-writes and sets lower values boundaries
        self._grid_cell_corners[0, :-1 :-1] =  self._grid[0] - self._grid_cell_size[0] / 2.0
        self._grid_cell_corners[1, :-1, :-1] = self._grid[1] - self._grid_cell_size[1] / 2.0

    def _update_grid_cells(self):

        for grid_cell in self._grid_cells:

            grid_cell.set_grid_coordinates(self._grid_cell_corners)

    def _init_pinning_matrix(self):

        pinning_matrix = self._pinning_matrix

        self._guess_grid_cell_size = self._APPROXIMATE_GRID_CELL_SIZES[
            pinning_matrix]

        self._grid = None
        self._grid_cell_size = None
        self._grid_cells = []
        self.features = {}

        for row in xrange(pinning_matrix[0]):

            for column in xrange(pinning_matrix[1]):

                if (not self._analysis_model.suppress_non_focal or
                        self._analysis_model.focus_position == (self._identifier[1], row, column)):

                    self._grid_cells.append(Grid_Cell(
                        [self._identifier, [row, column]], self._analysis_model))

    #
    # Get functions
    #

    def doAnalysis(self, im, image_model, save_grid_name=None):

        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        self._identifier[0] = image_model.index

        #Get an image-specific inter-scan-neutral transformation dictionary
        try:
            transpose_polynomial = imageBasics.Image_Transpose(
                sourceValues=image_model.grayscale_values,
                targetValues=image_model.grayscale_targets,
                polyCoeffs=self._polynomial_coeffs)
        except:
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

            grid_cell = self.grid_cells[(self._analysis_model.focus_position[1], self._analysis_model.focus_position[2])]

            self.watch_blob = grid_cell.get_item('blob').filter_array.copy()

            self.watch_source = grid_cell.get_item('blob').grid_array.copy()
            self.watch_results = self.features[grid_cell.position]