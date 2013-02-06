#!/usr/bin/env python
"""Part of analysis work-flow that holds a grid arrays"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
from scipy.optimize import fsolve
import os
import types
import sys
from matplotlib import pyplot as plt

#
# SCANNOMATIC LIBRARIES
#

import resource_grid
import analysis_grid_cell as grid_cell
import resource_logger as logger
import resource_path

#
# EXCEPTIONS

class Invalid_Grid(Exception): pass

#
# CLASS: Grid_Array
#


class Grid_Array():

    _APPROXIMATE_GRID_CELL_SIZES = {
        (8, 12): (212, 212),
        (16, 24): (106, 106),
        (32, 48): (53.64928854, 52.69155633),
        (64, 96): (27, 27),
        None: None
        }

    def __init__(self, parent, identifier, pinning_matrix, 
        verbose=False, visual=False, suppress_analysis=False,
        grid_array_settings=None, gridding_settings=None,
        grid_cell_settings=None):

        self._parent = parent

        if parent is None:

            self.logger = logger.Log_Garbage_Collector()
            self._paths = resource_path.Paths(
                src_path=os.path.abspath(__file__))
            self.fixture = None

        else:

            self.logger = self._parent.logger
            self._paths = parent._paths
            self.fixture = parent.fixture

        if type(identifier) == types.IntType:

            identifier = ("unknown", identifier)

        elif len(identifier) == 1:

            identifier = ["unknown", identifier[0]]

        else:

            identifier = [identifier[0], identifier[1]]

        self._identifier = identifier

        """
        self._analysis = array_dissection.Grid_Analysis(self, 
            pinning_matrix, verbose=verbose, visual=visual,
            gridding_settings=gridding_settings)
        """

        self.watch_source = None
        self.watch_scaled = None
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
                self.get_polynomial_coeffs()

        self.grid_cell_settings = grid_cell_settings

        self._guess_grid_cell_size = None
        self._grid_cell_size = None
        self._grid_cells = None
        self._grid = None


        self._features = []

        self.R = None
        self._first_analysis = True

        self._old_blob_img = None
        self._old_blob_filter = None
        self.track_times = []
        self.track_values = []
        self._old_timestamp = None

        self._pinning_matrix = pinning_matrix
        #if pinning_matrix != None:


        self._im_dim_order = None

        if pinning_matrix is not None:
            self._init_pinning_matrix()


    #
    # SET functions
    #

    def set_manual_ideal_grid(self, grid):

        
        best_fit_rows = grid[0]
        r = len(best_fit_rows)
        best_fit_columns = grid[1]
        c = len(best_fit_columns)

        X = np.array(best_fit_rows * c).reshape(c, r).T
        Y = np.array(best_fit_columns * r).reshape(r, c)
        self._grid = np.zeros((r, c, 2), dtype=np.float)
        self._grid[:,:,0] = X
        self._grid[:,:,1] = Y
        self._set_grid_cell_size()
        self.unset_history()

    def set_manual_grid(self, grid):

        self._grid = grid
        self._set_grid_cell_size()
        self.unset_history()

    def _set_grid_cell_size(self):

        dx = (self._grid[1:, :, 0] - self._grid[:-1, :, 0]).mean()
        dy = (self._grid[:, 1:, 1] - self._grid[:, :-1, 1]).mean()

        self._grid_cell_size = map(lambda x: int(round(x)), (dx, dy))

        """ Should be set based on im and not depend on im-orientation

        if self._im_dim_order is not None:
            self._grid_cell_size = [
                    self._grid_cell_size[self._im_dim_order[0]],
                    self._grid_cell_size[self._im_dim_order[1]]]

        """

    def unset_history(self):

        grid_history = self.fixture['history']

        p_uuid = self._parent.p_uuid
        plate = self._identifier[1]

        if p_uuid is not None:

            grid_history.unset_gridding_parameters(p_uuid,
                self._pinning_matrix, plate)


    def set_history(self, center, spacings):

        grid_history = self.fixture['history']

        p_uuid = self._parent.p_uuid
        plate = self._identifier[1]

        if p_uuid is not None:

            grid_history.set_gridding_parameters(p_uuid, self._pinning_matrix,
                plate, center, spacings)

    def set_grid(self, im, save_name=None, grid_correction=None):

        #Map so grid axis concur with image rotation
        if self._im_dim_order is None:

            self._im_dim_order = self._get_grid_to_im_axis_mapping(
                self._pinning_matrix, im)

        grid_shape = (self._pinning_matrix[int(self._im_dim_order[0])], 
                self._pinning_matrix[int(self._im_dim_order[1])])

        gh = np.array(self.get_history(self))

        if gh.size < 40:  # Require 10 projects (4 measures per project)
            gh_median = np.median(gh, axis=0)
            validate_parameters = True
            expected_spacings = tuple(gh_median[2:])
            expected_center = tuple(gh_median[:2])    
        else:
            #If too little reference data, use very rough guesses
            validate_parameters = False
            expected_spacings = self._guess_grid_cell_size
            expected_center = tuple([s/2.0 for s in im.shape])

        self._grid, X, Y, center, spacings, adjusted_values = \
                resource_grid.get_grid(im,
                expected_spacing=expected_spacings,
                expected_center=expected_center,
                grid_shape=grid_shape)

        if self._grid is None:
            return False

        if grid_correction is not None:
            self._grid -= grid_correction
            adjusted_values = True

        if self._grid.min() < 0:
            raise Invalid_Grid("Negative positons in grid")
            return False

        if (self._grid.shape[0] != self._pinning_matrix[self._im_dim_order[0]]
            and self._grid.shape[1] != 
            self._pinning_matrix[self._im_dim_order[1]]):

            raise Invalid_Grid(
                "Grid shape {0} missmatch with pinning matrix {1}".format(
                self._grid.shape,
                (self._pinning_matrix[self._im_dim_order[0]],
                self._pinning_matrix[self._im_dim_order[1]])))

            return False

        self._grid_cell_size = map(lambda x: int(round(x)), spacings)

        if adjusted_values:
            self.unset_history()
        else:
            self.set_history(center, spacings)

        if save_name is not None:
            self.make_grid_im(im, save_grid_name=save_name)

        if self.visual:

            self.make_grid_im(im)

        return True

    def make_grid_im(self, im, save_grid_name=None, X=None, Y=None):

        grid_image = plt.figure()
        grid_plot = grid_image.add_subplot(111)
        grid_plot.imshow(im, cmap=plt.cm.gray)
        ido = self._im_dim_order

        for row in xrange(self._grid.shape[0]):

            grid_plot.plot(
                self._grid[row,:,1],
                self._grid[row,:,0],
                'r-')

        for col in xrange(self._grid.shape[1]):

            grid_plot.plot(
                self._grid[:,col,1],
                self._grid[:,col,0],
                'r-')

        if X is not None and Y is not None:

            grid_plot.plot(Y,X, 'o', alpha=0.75,
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
            self.logger.info("ANALYSIS GRID: Saving grid-image as file" +\
                        " '{0}' for plate {1}".format(
                        save_grid_name, self._identifier[1]))

            grid_image.savefig(save_grid_name, pad_inches=0.01,
                format='svg', bbox_inches='tight')

            plt.close(grid_image)
            del grid_image

            self.logger.info("ANALYSIS GRID: Image saved!")

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

                self._grid_cells[row].append(grid_cell.Grid_Cell(\
                    self, [self._identifier,  [row, column]],
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

    def get_polynomial_coeffs(self):

        polynomial_coeffs = None

        try:


            fs = open(self._paths.analysis_polynomial, 'r')

        except:

            self.logger.critical("GRID ARRAY, " + \
                        "Cannot open polynomial info file")

            return None

        polynomial_coeffs = []

        for l in fs:

            l_data = eval(l.strip("\n"))

            if type(l_data) == types.ListType:

                polynomial_coeffs = l_data[-1]
                break

        fs.close()

        if polynomial_coeffs == []:

            polynomial_coeffs = None

        return polynomial_coeffs

    def get_p3(self, x):
        """
            returns the solution to:

                self.gs_a * x^3 + self.gs_b * x^2 + self.gs_c * x + self.gs_d

        """

        p = self.gs_a * (x ** 3) + self.gs_b * (x ** 2) + \
                self.gs_c * x + self.gs_d

        return p

    def get_transformation_matrix(self, gs_values=None, gs_fit=None,
                gs_indices=None, y_range=(0, 255), fix_axis=False):
        """get_transformation_matrix takes an coefficient array of a
        polynomial fit of the 3rd degree and calculates a matrix
        of all solutions for all the integer steps of the y-range
        specified.

        The function takes two arguments:

        @gs_values  A numpy array or a list of gray-scale values

        @gs_fit     A numpy array of the coefficients as returned
                    by numpy.polyfit, assuming 3rd degree
                    solution

        @gs_indices An optional list of gs indices if not a simple
                    enumerated range

        @y_range    A tuple having the including range limits
                    for the solution.

        @fix_axis   An optional possibility to fix the gs-axis,
                    else it will be made increasing (transformed with
                    -1 if not). Lowest value will also be set to 0,
                    assuming a continious series.

        The function returns a list of transformation values
        """

        if gs_values != None:

            if gs_indices == None:

                gs_indices = range(len(gs_values))

            if gs_indices[0] > gs_indices[-1]:

                gs_indices = map(lambda x: x * -1, gs_indices)

            if gs_indices[0] != 0:

                gs_indices = map(lambda x: x - gs_indices[0], gs_indices)

            tf_matrix = np.zeros((y_range[1] + 1))

            p = np.poly1d(np.polyfit(gs_indices, gs_values, 3))

            self.gs_a = p.c[0]
            self.gs_b = p.c[1]
            self.gs_c = p.c[2]
            self.gs_d = p.c[3]

            for i in xrange(256):

                #moving the line along y-axis
                self.gs_d = p.c[3] - i
                x = fsolve(self.get_p3, gs_values[0])

                #setting it back to get the values
                self.gs_d = p.c[3]
                tf_matrix[int(round(self.get_p3(x)))] = x

        else:

            tf_matrix = []

            for y in range(y_range[0], y_range[1] + 1):

                #Do something real here
                #The caluclated value shoud be a float

                x = float(y)
                tf_matrix.append(x)

        return tf_matrix

    def _get_grid_to_im_axis_mapping(self, pm, im):

        pm_max_pos = int(max(pm) == pm[1])
        im_max_pos = int(max(im.shape) == im.shape[1])

        im_axis_order = [int(pm_max_pos != im_max_pos)]
        im_axis_order.append(int(im_axis_order[0] == 0)) 

        self.logger.info("Axis order set to {0} based on pm {1} and im {2}".format(
            im_axis_order, pm, im.shape))

        return im_axis_order

    def _get_transformation_matrix_for_analysis(self, gs_values=None,
                gs_indices=None, gs_fit=None):

        #KODAK neutral scale
        if self._parent is not None:

            gs_indices = self._parent.gs_indices

        else:

            gs_indices = None

        if gs_values == None:

            transformation_matrix = self.get_transformation_matrix(\
                gs_fit=gs_fit, gs_indices=gs_indices)

        else:

            transformation_matrix = self.get_transformation_matrix(\
                gs_values=gs_values, gs_indices=gs_indices)

        return transformation_matrix

    def _set_tm_im(self, source, target, ul, tm, c_row, c_column):

        #wh          Width and Height
        wh = self._grid_cell_size

        source_view = source[ul[0]: ul[0] + wh[0], ul[1]: ul[1] + wh[1]]

        if source_view.shape != target.shape:

            raise Invalid_Grid(
                "Source view {0}, Target {1}, Source {2} ul {3}, wh {4}".format(
                source_view.shape, target.shape, source.shape, ul, wh))

        target[:,:] = tm[source_view]


    def get_analysis(self, im, gs_values=None, gs_fit=None, gs_indices=None,
            identifier_time=None, watch_colony=None, save_grid_name=None,
            grid_correction=None):

        #Resetting the values of the indepth watch colony
        self.watch_source = None
        self.watch_scaled = None
        self.watch_blob = None
        self.watch_results = None

        #Update info for future self-id reporting
        if identifier_time is not None:
            self._identifier[0] = identifier_time

        #Get an image-specific inter-scan-neutral transformation dictionary
        tm = self._get_transformation_matrix_for_analysis(gs_values=gs_values,
                    gs_fit=gs_fit, gs_indices=gs_indices)

        #Fast access to the pinning matrix
        pm = self._pinning_matrix


        #Only place grid if not yet placed
        if self._grid is None:

            if not self.set_grid(im):

                self.logger.critical('Failed to set grid on ' + \
                        '{0} and none to use'.format(self._identifier))

                return None

        im_dim_order = self._im_dim_order
        dim_reversed = im_dim_order[0] == 1

        #Save grid image if requested
        if save_grid_name is not None:

            self.make_grid_im(im, save_grid_name=save_grid_name)

        #Setting shortcuts for repeatedly used variable
        s_g = self._grid.copy()
        if grid_correction is not None:
            s_g += grid_correction  # Compensate if part of plate is outside im
        s_gcs = self._grid_cell_size
        s_g[:,:,0] -= s_gcs[0] / 2.0  # To get min-corner
        s_g[:,:,1] -= s_gcs[1] / 2.0  # To get min-corner
        l_d1 = pm[0]  # im_dim_order[0]]
        l_d2 = pm[1]  # im_dim_order[1]]

        #Setting up target array for trasnformation so it fits axis order
        tm_im = np.zeros(s_gcs, dtype=np.float64)

        #Go through the pinning abstract positions in order designated by pm
        for row in xrange(l_d1):

            for col in xrange(l_d2):

                #Only work on watched colonies if other's are suppressed
                if self.suppress_analysis == False or (watch_colony != None and \
                        watch_colony[1] == row and watch_colony[2] == col):

                    #Set up shortcuts
                    _cur_gc = self._grid_cells[row][col]
                    if dim_reversed:
                        row_min = s_g[col, row, 0]
                        col_min = s_g[col, row, 1]
                    else:
                        row_min = s_g[row, col, 0]
                        col_min = s_g[row, col, 1]

                    rc_min_tuple = (row_min, col_min)

                    #
                    #Finding the right part of the image
                    #-----------------------------------
                    #

                    """
                    #Set current gc center according to which pin we look at
                    _cur_gc.set_center(
                                    (rc_min_tuple[im_dim_order[0]],
                                    rc_min_tuple[im_dim_order[1]]),
                                    s_gcs)  # Does this do anything?
                    """

                    #
                    #Transforming to inter-scan neutal values
                    #----------------------------------------
                    #

                    #Set the tm_im for the region
                    if tm is not None:

                        self._set_tm_im(im, tm_im, rc_min_tuple, tm, row, col)

                    else:

                        #Shold make sure that tm_im is okay
                        self.logger.critical("ANALYSIS GRID ARRAY Lacks" + \
                                " transformation possibilities")

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
                            _cur_gc.get_analysis(no_analysis=True,
                            remember_filter=True)

                    #Info on the watched colony hooked up if that's the one
                    #analysed
                    if watch_colony != None:

                        if row == watch_colony[1] and \
                                    col == watch_colony[2]:

                            self.watch_blob = \
                                _cur_gc.get_item('blob').filter_array.copy()

                            self.watch_scaled = \
                                _cur_gc.get_item('blob').grid_array.copy()

                            self.watch_results = self._features[row][col]

        return self._features
