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
#from scipy.optimize import fsolve
import os
from matplotlib import pyplot as plt
import weakref

#
# SCANNOMATIC LIBRARIES
#

import grid
import grid_cell
import scanomatic.io.paths as paths
import imageBasics

#
# EXCEPTIONS


class Invalid_Grid(Exception):
    pass

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

    def __init__(self, parent, identifier, pinning_matrix,
                 visual=False, suppress_analysis=False,
                 grid_array_settings=None, gridding_settings=None,
                 grid_cell_settings=None):

        self._parent = weakref.ref(parent) if parent else None

        #self.logger = logging.getLogger("Grid Array {0}".format(identifier))

        if parent is None:

            self._paths = paths.Paths(
                src_path=os.path.abspath(__file__))
            self.fixture = None

        else:

            self._paths = parent._paths
            self.fixture = parent.fixture

        if isinstance(identifier, int):

            identifier = ("unknown", identifier)

        elif len(identifier) == 1:

            identifier = ["unknown", identifier[0]]

        else:

            identifier = [identifier[0], identifier[1]]

        self._identifier = identifier

        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        self._pinning_matrix = None

        default_settings = {'animate': False}

        if grid_array_settings is None:

            grid_array_settings = default_settings

        for k in default_settings.keys():

            if k in grid_array_settings.keys():

                setattr(self, k, grid_array_settings[k])

            else:

                setattr(self, k, default_settings[k])

        self.visual = visual
        self.suppress_analysis = suppress_analysis

        if grid_cell_settings is None:

            grid_cell_settings = dict()

        grid_cell_settings['polynomial_coeffs'] = \
            self.get_calibration_polynomial_coeffs()

        self.grid_cell_settings = grid_cell_settings

        self._guess_grid_cell_size = None
        self._grid_cell_size = None
        self._grid_cells = None
        self._grid = None

        self._features = []

        self._first_analysis = True

        self._pinning_matrix = pinning_matrix

        self._im_dim_order = None

        if pinning_matrix is not None:
            self._init_pinning_matrix()

    def __getitem__(self, key):
        return self._grid_cells[key[0]][key[1]]

    #
    # PROPERTIES
    #

    @property
    def features(self):
        return self._features

    @property
    def R(self):
        return None

    #
    # SET functions
    #

    """ Legacy methods, could possibly be useful 2013-08-22
    def set_manual_ideal_grid(self, grid):

        best_fit_rows = grid[0]
        r = len(best_fit_rows)
        best_fit_columns = grid[1]
        c = len(best_fit_columns)

        X = np.array(best_fit_rows * c).reshape(c, r).T
        Y = np.array(best_fit_columns * r).reshape(r, c)
        self._grid = np.zeros((2, r, c), dtype=np.float)
        self._grid[0, ...] = X
        self._grid[1, ...] = Y
        self._set_grid_cell_size()
        self.unset_history()

    def set_manual_grid(self, grid):

        self._grid = grid
        self._set_grid_cell_size()
        self.unset_history()
    """

    def _set_grid_cell_size(self):

        dx = (self._grid[0, 1:, :] - self._grid[0, :-1, :]).mean()
        dy = (self._grid[1, :, 1:] - self._grid[1, :, :-1]).mean()

        self._grid_cell_size = map(lambda x: int(round(x)), (dx, dy))

        """ Should be set based on im and not depend on im-orientation

        if self._im_dim_order is not None:
            self._grid_cell_size = [
                    self._grid_cell_size[self._im_dim_order[0]],
                    self._grid_cell_size[self._im_dim_order[1]]]

        """

    def unset_history(self):

        grid_history = self.fixture['history']

        p_uuid = self._parent().p_uuid
        plate = self._identifier[1]

        if p_uuid is not None:

            grid_history.unset_gridding_parameters(
                p_uuid, self._pinning_matrix, plate)

    def set_history(self, center, spacings):

        grid_history = self.fixture['history']

        p_uuid = self._parent().p_uuid
        plate = self._identifier[1]

        if p_uuid is not None:

            grid_history.set_gridding_parameters(
                p_uuid, self._pinning_matrix,
                plate, center, spacings)

        else:
            #self.logger.error("Gridding could not be saved because of no uuid")
            pass

    def set_grid(self, im, save_name=None, grid_correction=None):

        #Map so grid axis concur with image rotation
        if self._im_dim_order is None:

            self._im_dim_order = self._get_grid_to_im_axis_mapping(
                self._pinning_matrix, im)

        grid_shape = (self._pinning_matrix[int(self._im_dim_order[0])],
                      self._pinning_matrix[int(self._im_dim_order[1])])

        #self.logger.info("Setting a grid with format {0}".format(
        #    grid_shape))

        #gh = np.array(self.get_history())
        #self.logger.debug("Grid History {0}".format(gh))

        #If too little data, use very rough guesses
        if True:
            validate_parameters = False
            expected_spacings = self._guess_grid_cell_size
            expected_center = tuple([s / 2.0 for s in im.shape])
        """
        elif gh.size >= 40:  # Require 10 projects (4 measures per project)
            gh_median = np.median(gh, axis=0)
            validate_parameters = True
            expected_spacings = tuple(gh_median[2:])
            expected_center = tuple(gh_median[:2])
        if True:  # If some measures (3-9), use them
            validate_parameters = False  # But don't enforce
            gh_mean = np.mean(gh, axis=0)
            expected_spacings = tuple(gh_mean[2:])
            expected_center = tuple(gh_mean[:2])
        """

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
                self.make_grid_im(im, save_grid_name=save_name, grid=grd)

            return False

        if (self._grid.shape[1] != self._pinning_matrix[self._im_dim_order[0]]
                or self._grid.shape[2] !=
                self._pinning_matrix[self._im_dim_order[1]]):

            raise Invalid_Grid(
                "Grid shape {0} missmatch with pinning matrix {1}".format(
                self._grid.shape,
                (self._pinning_matrix[self._im_dim_order[0]],
                self._pinning_matrix[self._im_dim_order[1]])))

            return False

        #self.logger.info("Got center {0} and Spacings {1}".format(
        #    center, spacings))

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)

        if adjusted_values:
            #self.logger.info("Gridding got adjusted by history")
            self.unset_history()
        else:
            #self.logger.info("Setting gridding history")
            self.set_history(center, spacings)

        if save_name is not None:
            self.make_grid_im(im, save_grid_name=save_name)

        if self.visual:

            self.make_grid_im(im)

        return True

    def make_grid_im(self, im, save_grid_name=None, grid=None, X=None, Y=None):

        grid_image = plt.figure()
        grid_plot = grid_image.add_subplot(111)
        grid_plot.imshow(im, cmap=plt.cm.gray)

        if grid is None:
            grid = self._grid

        for row in xrange(grid.shape[1]):

            grid_plot.plot(
                grid[1, row, :],
                grid[0, row, :],
                'r-')

        for col in xrange(grid.shape[2]):

            grid_plot.plot(
                grid[1, :, col],
                grid[0, :, col],
                'r-')

        grid_plot.plot(grid[1, 0, 0],
                       grid[0, 0, 0],
                       'o', alpha=0.75, ms=10, mfc='none', mec='blue', mew=1)

        if X is not None and Y is not None:

            grid_plot.plot(Y, X, 'o', alpha=0.75,
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

            save_grid_name += "{0}.svg".format(self._identifier[1] + 1)
            #self.logger.info(
            #    "ANALYSIS GRID: Saving grid-image as file" +
            #    " '{0}' for plate {1}".format(
            #    save_grid_name, self._identifier[1]))

            grid_image.savefig(save_grid_name, pad_inches=0.01,
                               format='svg', bbox_inches='tight')

            grid_image.clf()
            plt.close(grid_image)
            del grid_image

            #self.logger.info("ANALYSIS GRID: Image saved!")

    def _init_pinning_matrix(self):
        """
            set_pinning_matrix sets the pinning_matrix.

            The function takes the following argument:

            @pinning_matrix  A list/tuple/array where first position is
                            the number of rows to be detected and second
                            is the number of columns to be detected.

        """
        pinning_matrix = self._pinning_matrix

        self._guess_grid_cell_size = self._APPROXIMATE_GRID_CELL_SIZES[
            pinning_matrix]

        self._grid = None
        self._grid_cell_size = None

        self._grid_cells = []
        self._features = []

        for row in xrange(pinning_matrix[0]):

            self._grid_cells.append([])
            self._features.append([])

            for column in xrange(pinning_matrix[1]):

                self._grid_cells[row].append(grid_cell.Grid_Cell(
                    [self._identifier,  [row, column]],
                    grid_cell_settings=self.grid_cell_settings))

                self._features[row].append(None)

    #
    # Get functions
    #

    def get_history(self):

        grid_history = self.fixture['history']

        plate = self._identifier[1]

        gh = grid_history.get_gridding_history(plate, self._pinning_matrix)

        return gh

    def get_calibration_polynomial_coeffs(self):

        polynomial_coeffs = None

        try:

            fs = open(self._paths.analysis_polynomial, 'r')

        except:

            #self.logger.critical(
            #    "GRID ARRAY, Cannot open polynomial info file")

            return None

        polynomial_coeffs = []

        for l in fs:

            l_data = eval(l.strip("\n"))

            if isinstance(l_data, list):

                polynomial_coeffs = l_data[-1]
                break

        fs.close()

        if polynomial_coeffs == []:

            polynomial_coeffs = None

        return polynomial_coeffs

    def _get_grid_to_im_axis_mapping(self, pm, im):

        pm_max_pos = int(max(pm) == pm[1])
        im_max_pos = int(max(im.shape) == im.shape[1])

        im_axis_order = [int(pm_max_pos != im_max_pos)]
        im_axis_order.append(int(im_axis_order[0] == 0))

        #self.logger.info("Axis order set to {0} based on pm {1} and im {2}".format(
        #    im_axis_order, pm, im.shape))

        return im_axis_order

    def _set_image_transposition(self, source, target, ul, wh,
                                 imTransposePoly):

        sourceView = source[ul[0]: ul[0] + wh[0], ul[1]: ul[1] + wh[1]]

        if sourceView.shape != target.shape:

            raise Invalid_Grid(
                "Grid Cell @ {0} has wrong size ({1} != {2})".format(
                    ul, sourceView.shape, target.shape))

        target[...] = imTransposePoly(sourceView)

    def doAnalysis(
            self, im, grayscaleSource=None, grayscaleTarget=None,
            grayscalePolyCoeffs=None,
            identifier_time=None, watch_colony=None,
            save_grid_name=None, grid_correction=None):

        #Resetting the values of the indepth watch colony
        self.watch_source = None
        self.watch_blob = None
        self.watch_results = None

        #Update info for future self-id reporting
        if identifier_time is not None:
            self._identifier[0] = identifier_time

        #Get an image-specific inter-scan-neutral transformation dictionary
        try:
            transposePoly = imageBasics.Image_Transpose(
                sourceValues=grayscaleSource,
                targetValues=grayscaleTarget,
                polyCoeffs=grayscalePolyCoeffs)
        except:
            transposePoly = None

        #Fast access to the pinning matrix
        pm = self._pinning_matrix

        #Only place grid if not yet placed
        if self._grid is None:

            if not self.set_grid(im):

                #self.logger.critical(
                #    'Failed to set grid on ' +
                #    '{0} and none to use'.format(self._identifier))

                return None

        im_dim_order = self._im_dim_order
        dim_reversed = im_dim_order[0] == 1

        #Save grid image if requested
        if save_grid_name is not None:

            self.make_grid_im(im, save_grid_name=save_grid_name)

        #Setting shortcuts for repeatedly used variable
        s_g = self._grid.copy()
        s_gcs = self._grid_cell_size
        s_g[0, ...] -= s_gcs[0] / 2.0  # To get min-corner
        s_g[1, ...] -= s_gcs[1] / 2.0  # To get min-corner
        l_d1 = pm[0]  # im_dim_order[0]]
        l_d2 = pm[1]  # im_dim_order[1]]

        #Setting up target array for trasnformation so it fits axis order
        tm_im = np.zeros(s_gcs, dtype=np.float64)

        #Go through the pinning abstract positions in order designated by pm
        for row in xrange(l_d1):

            for col in xrange(l_d2):

                #Only work on watched colonies if other's are suppressed
                if (self.suppress_analysis is False or
                        (watch_colony is not None and
                         watch_colony[1] == row and watch_colony[2] == col)):

                    #Set up shortcuts
                    _cur_gc = self._grid_cells[row][col]
                    if dim_reversed:
                        row_min = s_g[0, col, row]
                        col_min = s_g[1, col, row]
                    else:
                        row_min = s_g[0, row, col]
                        col_min = s_g[1, row, col]

                    rc_min_tuple = (row_min, col_min)

                    #
                    #Transforming to inter-scan neutal values
                    #----------------------------------------
                    #

                    #Set the tm_im for the region
                    if transposePoly is not None:

                        #self._set_tm_im(im, tm_im, rc_min_tuple, tm, row, col)
                        self._set_image_transposition(
                            im, tm_im, rc_min_tuple, s_gcs, transposePoly)

                    else:

                        #Shold make sure that tm_im is okay
                        #self.logger.critical("ANALYSIS GRID ARRAY Lacks" +
                        #                     " transformation possibilities")
                        pass

                    #
                    #Setting up the grid cell
                    #------------------------
                    #

                    #Sets the tm_im as the gc data source
                    _cur_gc.set_data_source(tm_im)

                    #This happens only the first time, setting up the analysis
                    #layers of the grid cell

                    if self._first_analysis:

                        _cur_gc.attach_analysis(
                            blob=True, background=True, cell=True,
                            run_detect=False)

                    #
                    #Getting the analysis for all layers of the Grid Cell
                    #----------------------------------------------------
                    #

                    self._features[row][col] = \
                        _cur_gc.get_analysis(remember_filter=True)

                    #Info on the watched colony hooked up if that's the one
                    #analysed
                    if watch_colony is not None:

                        if (row == watch_colony[1] and
                                col == watch_colony[2]):

                            self.watch_blob = \
                                _cur_gc.get_item('blob').filter_array.copy()

                            self.watch_source = \
                                _cur_gc.get_item('blob').grid_array.copy()

                            self.watch_results = self._features[row][col]

        return self._features
