#!/usr/bin/env python
"""Part of analysis work-flow that holds a grid arrays"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.997"
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

import analysis_grid_array_dissection as array_dissection
import resource_grid
import analysis_grid_cell as grid_cell
import resource_logger as logger

#
# CLASS: Grid_Array
#


class Grid_Array():

    _APPROXIMATE_GRID_CELL_SIZES = {
        (8, 12): (212, 212),
        (16, 24): (106, 106),
        (32, 48): (54, 54),
        (64, 96): (27, 27),
        None: None
        }

    def __init__(self, parent, identifier, pinning_matrix, 
        verbose=False, visual=False, suppress_analysis=False,
        grid_array_settings=None, gridding_settings=None,
        grid_cell_settings=None):

        self._parent = parent
        self.fixture = parent.fixture

        if parent is None:

            self.logger = logger.Log_Garbage_Collector()
            self._paths = resource_path.Path(
                src_path=os.path.abspath(__file__))

        else:

            self.logger = self._parent.logger
            self._paths = parent._paths

        if type(identifier) == types.IntType:

            identifier = ("unknown", identifier)

        elif len(identifier) == 1:

            identifier = ["unknown", identifier[0]]

        else:

            identifier = [identifier[0], identifier[1]]

        self._identifier = identifier


        self._analysis = array_dissection.Grid_Analysis(self, 
            pinning_matrix, verbose=verbose, visual=visual,
            gridding_settings=gridding_settings)

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

        if pinning_matrix != None:

            self._set_pinning_matrix(pinning_matrix)

        self._im_dim_order = None


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

    def set_manual_grid(self, grid):

        self._grid = grid
        self._set_grid_cell_size()

    def _set_grid_cell_size(self):

        dx = (self._grid[1:, :, 0] - self._grid[:-1, :, 0]).mean()
        dy = (self._grid[:, 1:, 1] - self._grid[:, :-1, 1]).mean()

        self._grid_cell_size = map(lambda x: int(round(x)), (dx, dy))

        if self._im_dim_order is not None:
            self._grid_cell_size = [
                    self._grid_cell_size[self._im_dim_order[0]],
                    self._grid_cell_size[self._im_dim_order[1]]]

        if self._parent is not None:

            self.set_history()

    def set_history(self, adjusted_by_history=False):

        topleft_history = self.fixture.get_pinning_history(
                self._identifier[1], self._pinning_matrix)

        if topleft_history is None:

            topleft_history = []

        p_uuid = self._parent.p_uuid

        if p_uuid is not None and not adjusted_by_history:

            is_rerun = [i for i, tl in enumerate(topleft_history) \
                        if tl[0] == p_uuid]


            hist_entry = (p_uuid,
                    self._grid[0,0,:].tolist(),
                    self._grid_cell_size)

            if len(is_rerun) == 0:
                topleft_history.append(hist_entry)
            else:
                topleft_history[is_rerun[0]] = hist_entry

            if len(topleft_history) > 20:
                del topleft_history[0]

            self.fixture.set_pinning_positions(\
                self._identifier[1], self._pinning_matrix, topleft_history)

    def set_grid(self, im):

        self._grid = resource_grid.get_grid(im, box_size=self._guess_grid_cell_size, 
                grid_shape=self._pinning_matrix)

        self._set_grid_cell_size()

        if self.visual:

            self.make_grid_im(im)

        return True

    def make_grid_im(self, im, save_grid_name=None):

        best_fit_columns = self._grid.mean(0)
        best_fit_rows = self._grid.mean(1)

        grid_image = plt.figure()
        grid_plot = grid_image.add_subplot(111)
        grid_plot.imshow(im, cmap=plt.cm.gray)

        for row in xrange(self._pinning_matrix[0]):

            if self._im_dim_order[0] == 1:

                grid_plot.plot(
                        np.ones(best_fit_columns.size) * \
                        best_fit_rows[row],
                        best_fit_columns,
                        'r-')

            else:

                grid_plot.plot(
                        best_fit_columns,
                        np.ones(best_fit_columns.size) * \
                        best_fit_rows[row],
                        'r-')

            for column in xrange(self._pinning_matrix[1]):

                if self._im_dim_order[0] == 1:

                    grid_plot.plot(
                            best_fit_rows,
                            np.ones(best_fit_rows.size) * \
                            best_fit_columns[column],
                            'r-')

                else:

                    grid_plot.plot(
                            np.ones(best_fit_rows.size) * \
                            best_fit_columns[column],
                            best_fit_rows,
                            'r-')

        ax = grid_image.gca()
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(0, im.shape[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if save_grid_name is None:

            grid_image.show()
            plt.close(grid_image)
            del grid_image

        else:

            save_grid_name += "{0}.png".format(self._identifier[1])
            self.logger.info("ANALYSIS GRID: Saving grid-image as file" +\
                        " '{0}' for plate {1}".format(
                        save_grid_name, self._identifier[1]))

            grid_image.savefig(save_grid_name, pad_inches=0.01,
                                bbox_inches='tight')

            plt.close(grid_image)
            del grid_image

            self.logger.info("ANALYSIS GRID: Image saved!")

    def _set_pinning_matrix(self, pinning_matrix):
        """
            set_pinning_matrix sets the pinning_matrix.

            The function takes the following argument:

            @pinning_matrix  A list/tuple/array where first position is
                            the number of rows to be detected and second
                            is the number of columns to be detected.

        """

        self._pinning_matrix = pinning_matrix

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
        """Places the transformed image values in target

        Used by algorithm:
        ------------------

        source      image that has the section that should be transformed
        target      image with size matching wh (see below) that will take the
                    transformed im
        ul          Upper Left corner on source
        tm          Transformation matrix

        Used only for error reporting:
        ------------------------------

        c_row       The row on the pinning matrix (dim 0 of pm) currently used
        c_column    The column / (dim 1)
        """

        #Clear target of might just be laying there
        target *= 0
    
        #wh          Width and Height
        wh = self._grid_cell_size

        #axis_order  Tells the orientation of the grid on the source
        axis_order = self._im_dim_order

        #There's probably some faster way
        self.logger.debug("ANALYSIS GRID ARRAY Transforming -> Kodak")

        #Creating shortcut for repeaded lookup
        im_axis_order = self._im_dim_order

        #Iteration direction priority is decided by how the pinning matrix
        #was written and not how the image looks

        #x,y        are target coordinates
        #x2, y2     are source coordinates

        wh_half = [i/2.0 for i in wh]

        for x in xrange(wh[im_axis_order[0]]):

            x2 = int(round(ul[axis_order[0]] - 
                    wh_half[im_axis_order[0]])) + x

            for y in xrange(wh[axis_order[1]]):

                y2 = int(round(ul[axis_order[1]] -
                    wh_half[im_axis_order[1]])) + y

                try:

                    #This makes sure that the target and source have the
                    #same rotation.

                    #Common logic in words: "For each pixel, in source, look
                    #up the transformed value in the tm and place that value
                    #in the corresponting positon in target

                    if axis_order[0] > axis_order[1]:

                        target[y, x] = tm[source[x2, y2]]

                    else:

                        target[x, y] = tm[source[x2, y2]]
                        #print target[x, y], source[x2, y2], tm[source[x2, y2]]

                except IndexError:

                    err_str = (
                        "Index Error:An image has been " + 
                        " saved as gridding_error.png\n" + 
                        "target.shape {0} stopped at ({1}, {2})".format(
                        (target.shape[axis_order[0]],
                        target.shape[axis_order[1]]), x, y) + 
                        ("which corresponds to  ({0}, {1}) on source" +
                        " (shape {2})").format(x2, y2, 
                        (source.shape[axis_order[0]],
                        source.shape[axis_order[1]])) + 
                        ("Origin on source was: ({0}, {1}) and attempted" +
                        " size was ({2}, {3}) ").format(
                        ul[axis_order[0]], ul[axis_order[1]],
                        wh[im_axis_order[0]], wh[im_axis_order[1]]) +
                        "from {0}:{1}:{2}".format(
                        self._identifier, (c_row, self._pinning_matrix[0]),
                        (c_column, self._pinning_matrix[1])))

                    grid_image = plt.figure()
                    grid_plot = grid_image.add_subplot(111)

                    if axis_order[0] > axis_order[1]:

                        grid_plot.imshow(source.T, cmap=plt.cm.Greys)

                    else:

                        grid_plot.imshow(source, cmap=plt.cm.Greys)

                    grid_plot.set_xlim(0, source.shape[axis_order[0]])
                    grid_plot.set_ylim(0, source.shape[axis_order[1]])

                    best_fit_columns = self._grid.mean(0)
                    best_fit_rows = self._grid.mean(1)

                    for row in xrange(self._pinning_matrix[0]):

                        grid_plot.plot(
                            np.ones(
                            best_fit_columns.size) * \
                            best_fit_rows[row],
                            best_fit_columns,
                            'r-')

                        for column in xrange(
                                self._pinning_matrix[1]):

                            grid_plot.plot(
                                best_fit_rows,
                                np.ones(best_fit_rows.size) * \
                                best_fit_columns[column],
                                'r-')

                    grid_plot.add_patch(plt.Rectangle((x2,y2),
                        wh[axis_order[0]],
                        wh[axis_order[1]],
                        ls='solid', lw=2,
                        fill=False, ec=(0.9, 0.9, .1, 1)))

                    grid_image.savefig("gridding_error.png")

                    raise Exception(IndexError, err_str)

                    sys.exit()

    def get_analysis(self, im, gs_values=None, gs_fit=None, gs_indices=None,
            identifier_time=None, watch_colony=None, save_grid_name=None):

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

        #Map so grid axis concur with image rotation
        if self._im_dim_order is None:

            self._im_dim_order = self._get_grid_to_im_axis_mapping(pm, im)

            self._analysis.set_dim_order(self._im_dim_order)

        im_dim_order = self._im_dim_order

        #Only place grid if not yet placed
        if self._best_fit_columns is None:

            if not self.set_grid(im):

                self.logger.critical('Failed to set grid on ' + \
                        '{0} and none to use'.format(self._identifier))

                return None

        #Save grid image if requested
        if save_grid_name is not None:

            self.make_grid_im(im, save_grid_name=save_grid_name)

        #Setting shortcuts for repeatedly used variable
        s_g = self._grid.copy()
        s_gcs = self._grid_cell_size
        s_g[:,:,0] -= s_gcs[0] / 2.0  # To get min-corner
        s_g[:,:,1] -= s_gcs[1] / 2.0  # To get min-corner

        #Setting up target array for trasnformation so it fits axis order
        tm_im = np.zeros(s_gcs, dtype=np.float64)

        #Go through the pinning abstract positions in order designated by pm
        for row in xrange(pm[0]):

            for col in xrange(pm[1]):

                #Only work on watched colonies if other's are suppressed
                if self.suppress_analysis == False or (watch_colony != None and \
                        watch_colony[1] == row and watch_colony[2] == col):

                    #Set up shortcuts
                    _cur_gc = self._grid_cells[row][col]

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
