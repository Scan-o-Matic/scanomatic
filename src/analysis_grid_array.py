#!/usr/bin/env python
"""Part of analysis work-flow that holds a grid arrays"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.996"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
#import logging
from scipy.optimize import fsolve
import os
import types
import sys
from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec

#
# SCANNOMATIC LIBRARIES
#

import analysis_grid_array_dissection as array_dissection
import analysis_grid_cell as grid_cell
import resource_logger as logger

#
# CLASS: Grid_Array
#


class Grid_Array():

    def __init__(self, parent, identifier, pinning_matrix, 
        use_otsu=True, median_coeff=0.99, verbose=False,
        visual=False, manual_threshold=0.05, suppress_other=False,
        use_fallback_detection=False):

        self._parent = parent

        if parent is None:

            self.logger = logger.Log_Garbage_Collector()

        else:

            self.logger = self._parent.logger

        if type(identifier) == types.IntType:

            identifier = ("unknown", identifier)

        elif len(identifier) == 1:

            identifier = ["unknown", identifier[0]]

        else:

            identifier = [identifier[0], identifier[1]]

        self._identifier = identifier

        self._analysis = array_dissection.Grid_Analysis(self, 
            pinning_matrix, use_otsu=use_otsu,
            median_coeff=median_coeff, verbose=verbose, visual=visual,
            manual_threshold=manual_threshold)

        self.watch_source = None
        self.watch_scaled = None
        self.watch_blob = None
        self.watch_results = None

        self._pinning_matrix = None
        self.suppress_other = suppress_other
        self.use_fallback_detection = use_fallback_detection
        self.visual = visual

        self._grid_cell_size = None
        self._grid_cells = None

        self._features = []

        self.R = None
        self._best_fit_rows = None
        self._best_fit_columns = None
        self._first_analysis = True

        self._old_blob_img = None
        self._old_blob_filter = None
        self.track_times = []
        self.track_values = []
        self._old_timestamp = None

        if pinning_matrix != None:

            self._set_pinning_matrix(pinning_matrix)

        self._polynomial_coeffs = None

        self._im_dim_order = None

        if parent is not None:

            self._config_calibration_polynomial = \
                            parent._program_config_root +\
                            os.sep + "calibration.polynomials"

            get_poly = True

            try:

                fs = open(self._config_calibration_polynomial, 'r')

            except:

                self.logger.critical("GRID ARRAY, " + \
                            "Cannot open polynomial info file")

                get_poly = False

            if get_poly:

                self._polynomial_coeffs = []

                for l in fs:

                    l_data = eval(l.strip("\n"))

                    if type(l_data) == types.ListType:

                        self._polynomial_coeffs = l_data[-1]
                        break

                fs.close()

                if self._polynomial_coeffs == []:

                    self._polynomial_coeffs = None

    #
    # SET functions
    #

    def set_manual_grid(self, grid):

        best_fit_rows = grid[0]
        best_fit_columns = grid[1]

        self._best_fit_rows = best_fit_rows
        self._best_fit_columns = best_fit_columns

        best_fit_rows = np.asarray(best_fit_rows)
        best_fit_columns = np.asarray(best_fit_columns)

        fit_frequency = ((best_fit_columns[1:] - best_fit_columns[:-1]).mean(),
                        (best_fit_rows[1:] - best_fit_rows[:-1]).mean())

        if self._grid_cell_size is None:

            self._grid_cell_size = map(int, map(round, fit_frequency[:]))

        if self._im_dim_order is not None:
            self._grid_cell_size = [
                    self._grid_cell_size[self._im_dim_order[0]],
                    self._grid_cell_size[self._im_dim_order[1]]]

        if self._parent is not None:

            self.set_history()

    def set_history(self, adjusted_by_history=False):

        topleft_history = self._parent.fixture.get_pinning_history(
                self._identifier[1], self._pinning_matrix)

        if topleft_history is None:

            topleft_history = []

        p_uuid = self._parent.p_uuid

        if p_uuid is not None:

            is_rerun = [i for i, tl in enumerate(topleft_history) \
                        if tl[0] == p_uuid]

            hist_b_r = np.asarray(self._best_fit_rows)
            hist_b_c = np.asarray(self._best_fit_columns)

            hist_entry = (p_uuid,
                    (self._best_fit_rows[0], self._best_fit_columns[0]),
                    ((hist_b_r[1:] - hist_b_r[: -1]).mean(),
                    (hist_b_c[1:] - hist_b_c[: -1]).mean()))

            if not adjusted_by_history:
                if len(is_rerun) == 0:
                    topleft_history.append(hist_entry)
                else:
                    topleft_history[is_rerun[0]] = hist_entry

                if len(topleft_history) > 20:
                    del topleft_history[0]

                self._parent.fixture.set_pinning_positions(\
                    self._identifier[1], self._pinning_matrix, topleft_history)

    def set_grid(self, im):
        """Sets a grid to an image.

        @param im: An array / the image

        @param use_otsu : Causes thresholding to be done by Otsu
        algorithm (Default)

        @param median_coeff : Coefficient to threshold from the
        median when not using Otsu.

        @param save_grid_name : A custom name for the saved image,
        if left out it is not saved.

        @param verbose : If a lot of things should be printed out

        @param visual : If visual information should be presented.

        The function returns True if it did set the grid.
        """

        best_fit_rows = self._best_fit_rows
        best_fit_columns = self._best_fit_columns

        if self._parent is not None:

            topleft_history = self._parent.fixture.get_pinning_history(
                        self._identifier[1], self._pinning_matrix)

        else:

            topleft_history = []

        if topleft_history is None:

            topleft_history = []

        self.logger.debug('The gridding history is {0}'.format(
                    topleft_history))

        best_fit_rows, best_fit_columns, R, adjusted_by_history = \
                self._analysis.get_analysis(im, 
                history=topleft_history)

        self.logger.debug("GRID ARRAY " + \
                "{0}, best rows \n{1}\nbest columns\n{2}".format(
                "unkown", best_fit_rows, best_fit_columns))

        #print "\n",best_fit_rows, best_fit_columns,"\n"

        if best_fit_rows == None or best_fit_columns == None:

            self.logger.warning("GRID ARRAY " + \
                        " {0}, Failed to detect grid.".format(
                        self._identifier))

            self._best_fit_rows = None
            self._best_fit_columns = None

            return False

        elif self.R is None or R < 20:

            self._best_fit_rows = best_fit_rows
            self._best_fit_columns = best_fit_columns

            if self._parent is not None:

                self.set_history(
                    adjusted_by_history=adjusted_by_history)

        else:

            logger.warning('Pinning matrix seem inconsistent with ' +\
                    'previous matrices in this project, using old')

        self.R = R

        if self._grid_cell_size is None:

            if self._analysis.best_fit_frequency is None:

                self.logger.critical("GRID ARRAY, No grid cell size" +
                        " obtained, probably because fixure calibration" +
                        "is bad.")

                return None

            self._grid_cell_size = map(int, map(round,
                        self._analysis.best_fit_frequency[:]))

            #print self._grid_cell_size

        if self.visual:

            self.make_grid_im(im)

        if self._best_fit_rows is not None:

            return True

        else:

            return False

    def make_grid_im(self, im, save_grid_name=None):

        best_fit_columns = self._best_fit_columns
        best_fit_rows = self._best_fit_rows

        grid_image = plt.figure()
        grid_plot = grid_image.add_subplot(111)
        grid_plot.imshow(im, cmap=plt.cm.gray)


        for row in xrange(self._pinning_matrix[0]):

            grid_plot.plot(\
                    np.ones(len(best_fit_columns)) * \
                    best_fit_rows[row],
                    np.array(best_fit_columns),
                    'r-')

            for column in xrange(self._pinning_matrix[1]):

                grid_plot.plot(\
                        np.array(best_fit_rows),
                        np.ones(len(best_fit_rows)) * \
                        best_fit_columns[column], 'r-')

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

    def _set_pinning_matrix(self, pinning_matrix):
        """
            set_pinning_matrix sets the pinning_matrix.

            The function takes the following argument:

            @pinning_matrix  A list/tuple/array where first position is
                            the number of rows to be detected and second
                            is the number of columns to be detected.

        """

        self._pinning_matrix = pinning_matrix

        self._grid_cells = []
        self._features = []

        for row in xrange(pinning_matrix[0]):

            self._grid_cells.append([])
            self._features.append([])

            for column in xrange(pinning_matrix[1]):
                self._grid_cells[row].append(grid_cell.Grid_Cell(\
                    self, [self._identifier,  [row, column]]))
                self._features[row].append(None)

    #
    # Get functions
    #

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

        ###DEBUG TF-matrix
        #print "Transformation matrix:\n", list(tf_matrix)
        ###END DEBUG

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

        #Clear source of might just be laying there
        source *= 0
    
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

        for x in xrange(wh[im_axis_order[0]]):

            x2 = int(round(ul[axis_order[0]])) + x
                        #- s_gcs[0] / 2.0)) + x

            for y in xrange(wh[axis_order[1]]):

                y2 = int(round(ul[axis_order[1]])) + y
                        #- s_gcs[1] / 2.0)) + y

                try:

                    #This makes sure that the target and source have the
                    #same rotation.

                    #Common logic in words: "For each pixel, in source, look
                    #up the transformed value in the tm and place that value
                    #in the corresponting positon in target

                    if axis_order[0] < axis_order[1]:

                        target[x, y] = tm[source[x2, y2]]

                    else:

                        target[y, x] = tm[source[y2, x2]]

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

                    for row in xrange(self._pinning_matrix[axis_order[0]]):

                        grid_plot.plot(
                            np.ones(
                            len(self._best_fit_columns)) * \
                            self._best_fit_rows[row],
                            np.array(self._best_fit_columns),
                            'r-')

                        for column in xrange(
                                self._pinning_matrix[axis_order[1]]):

                            grid_plot.plot(
                                np.array(self._best_fit_rows),
                                np.ones(
                                len(self._best_fit_rows)) * \
                                self._best_fit_columns[column],
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
        s_bfr = self._best_fit_rows
        s_bfc = self._best_fit_columns
        s_gcs = self._grid_cell_size

        #Setting up target array for trasnformation so it fits axis order
        tm_im = np.zeros(s_gcs, dtype=np.float64)

        #Go through the pinning abstract positions in order designated by pm
        for row in xrange(pm[0]):

            for col in xrange(pm[1]):

                #Only work on watched colonies if other's are suppressed
                if self.suppress_other == False or (watch_colony != None and \
                        watch_colony[1] == row and watch_colony[2] == col):

                    #Set up shortcuts
                    _cur_gc = self._grid_cells[row][col]

                    row_min = s_bfr[row]
                    col_min = s_bfc[col]
                    rc_min_tuple = (row_min, col_min)

                    #
                    #Finding the right part of the image
                    #-----------------------------------
                    #

                    #Set current gc center according to which pin we look at
                    _cur_gc.set_center(
                                    (rc_min_tuple[im_dim_order[0]],
                                    rc_min_tuple[im_dim_order[1]]),
                                    s_gcs)  # Does this do anything?

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
                                use_fallback_detection=self.use_fallback_detection,
                                run_detect=False)

                    #
                    #Detecting the features
                    #----------------------
                    #

                    #This step only detects the objects, but doesn't run 
                    #analysis. Reason: We need the background mean to do the
                    #final transformation to cell count space.

                    _cur_gc.get_analysis(no_analysis=True,
                                    remember_filter=True)

                    #
                    #Transfer data to 'Cell Estimate Space'
                    #--------------------------------------
                    #

                    bg_filter = _cur_gc.get_item('background').filter_array

                    #Check so that there actually was a background detected
                    #if not, something is horribly wrong and current measure
                    #should be skipped

                    #
                    #Reporting the findings
                    #----------------------
                    #

                    if bg_filter.sum() == 0:

                        self.logger.warning('Time/Plate ' + 
                                '{0}, Row: {1}, Column: {2}'.format(
                                self._identifier, row, col) + 
                                ' has no background (skipping)')

                        self._features[row][col] = None

                    else:

                        #Moving the pixel values to cell estimate space

                        _cur_gc.set_new_data_source_space(
                                space='cell estimate', bg_sub_source=bg_filter,
                                polynomial_coeffs=self._polynomial_coeffs)

                        #Just run the actual analysis on what was detected
                        #above
                        self._features[row][col] = \
                                _cur_gc.get_analysis(no_detect=True)

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

    def get_analysis_old(self, im, gs_fit=None, gs_values=None, use_fallback=False,
                use_otsu=True, median_coeff=None, verbose=False, visual=False,
                watch_colony=None, suppress_other=False,
                save_grid_name=None, identifier_time=None,
                save_anime_name=None, timestamp=None, animate=False):

        """Returns analysis!
        @param im: An array / the image

        @param gs_fit : An array of the fitted coefficients for the grayscale

        @param gs_values : An array of the grayscale pixelvalues, if
        submittet gs_fit is disregarded

        @param use_otsu : Causes thresholding to be done by Otsu
        algorithm (Default)

        @param median_coeff : Coefficient to threshold from the
        median when not using Otsu.

        @param verbose : If a lot of things should be printed out

        @param visual : If visual information should be presented.

        @param save_grid_name : A custom name for the saved image, if none
        is submitted, it will be grid.png in current directory.

        @param identifier_time : A time index to update the identifier with

        @param save_anime_name : Path to where to save animation figure

        The function returns two arrays, one per dimension, of the
        positions of the spikes and a quality index
        """

        if identifier_time is not None:
            self._identifier[0] = identifier_time

        debug_per_plate = False

        #DEBUGHACK
        #visual = True
        #verbose = True
        #debug_per_plate = True
        #DEBUGHACK - END

        self.watch_source = None
        self.watch_scaled = None
        self.watch_blob = None
        self.watch_results = None

        if debug_per_plate:

            raw_input("Waiting to start next plate (press Enter)")

        if self._best_fit_columns is None:

            if not self.set_grid(im, save_grid_name=save_grid_name,
                    use_otsu=use_otsu, median_coeff=median_coeff,
                    verbose=verbose, visual=visual):

                self.logger.critical('Failed to set grid on ' + \
                        '{0} and none to use'.format(self._identifier))

                return None

        if (min(self._pinning_matrix) == self._pinning_matrix[1]) != \
            (min(im.shape) == im.shape[1]):

            ax1 = 0
            ax2 = 1
            s_bfc = self._best_fit_rows
            s_bfr = self._best_fit_columns
            s_gcs = [self._grid_cell_size[1], self._grid_cell_size[0]]

        else:

            ax1 = 1
            ax2 = 0
            s_bfr = self._best_fit_rows
            s_bfc = self._best_fit_columns
            s_gcs = self._grid_cell_size

        #total_steps = float(self._pinning_matrix[0] * self._pinning_matrix[1])

        #DEBUG PLOT GRID
        #debug_fig = plt.figure()
        #debug_ax = debug_fig.add_subplot(1,1,1)
        #debug_ax.imshow(im)
        #if self._best_fit_rows is not None:
            #debug_cols = (0, im.shape[0])
            #for debug_row in self._best_fit_rows:
                #debug_ax.plot((debug_row, debug_row),debug_cols,'k')
        #if self._best_fit_columns is not None:
            #debug_rows = (0, im.shape[1])
            #for debug_col in self._best_fit_columns:
                #debug_ax.plot(debug_rows,(debug_col,debug_col),'k')
        #debug_fig.show()
        #self.logger.warning("Pining (R,C) " + \
            #"{0} best_rows {1} best_cols {2}".format(
            #self._pinning_matrix, self._best_fit_rows, best_fit_columns))
        #raw_input("> ")
        #DEBUG PLOT GRID END

        #Normalising towards grayscale before anything is done on the colonies
        transformation_matrix = None
        

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

        #print "\n***Transformation matrix"
        #print transformation_matrix

        #if watch_colony != None:
        #    ul = self._grid_cells[watch_colony[1]][watch_colony[2]]\
                #.get_top_left()
        #    lr = self._grid_cells[watch_colony[1]][watch_colony[2]]\
                #.get_bottom_right()
        #    self.watch_source = im[ul[1]:lr[1],ul[0]:lr[0]]

        #if transformation_matrix != None:
            #There's probably some faster way
            #for x in xrange(im.shape[0]):
                #for y in xrange(im.shape[1]):
                    #im[x,y] = transformation_matrix[im[x,y]]
        #print "*** Analysing grid:"

        tf_im = np.zeros(s_gcs, dtype=np.float64)

        for row in xrange(self._pinning_matrix[ax1]):

            for column in xrange(self._pinning_matrix[ax2]):

                if suppress_other == False or (watch_colony != None and \
                        watch_colony[1] == row and watch_colony[2] == column):

                    _cur_gc = self._grid_cells[row][column]

                    row_min = s_bfr[row]
                    col_min = s_bfc[column]

                    _cur_gc.set_center(
                                    (row_min,
                                    col_min),
                                    s_gcs)

                    if transformation_matrix is not None:

                        #There's probably some faster way
                        self.logger.debug(
                                "ANALYSIS GRID ARRAY Transforming -> Kodak")

                        for x in xrange(tf_im.shape[ax1]):

                            x2 = int(round(row_min)) + x
                                        #- s_gcs[0] / 2.0)) + x

                            for y in xrange(tf_im.shape[ax2]):

                                y2 = int(round(col_min)) + y
                                        #- s_gcs[1] / 2.0)) + y

                                try:
                                    tf_im[x, y] = \
                                        transformation_matrix[im[x2, y2]]

                                except IndexError:

                                    self.logger.critical(\
                                        "Index Error:An image has been " + \
                                        " saved as gridding_error.png\n" + \
                                        "tf_im.shape {0} vs ({1}, {2})".format(
                                        tf_im.shape, x, y) + \
                                        "and im.shape {0} vs ".format(im.shape
                                        ) + "({0}, {1})\nbest_fit ".format(
                                        x2, y2) + \
                                        "({0}, {1}) size ({2}, {3}) ".format(
                                        col_min,
                                        row_min,
                                        s_gcs[ax1],
                                        s_gcs[ax2]) + \
                                        "from {0}:{1}:{2}".format(
                                        self._identifier, (row, 
                                        self._pinning_matrix[ax1]), (column,
                                        self._pinning_matrix[ax2])))

                                    grid_image = plt.figure()
                                    grid_plot = grid_image.add_subplot(111)
                                    grid_plot.imshow(im, cmap=plt.cm.Greys)
                                    grid_plot.set_xlim(0, im.shape[ax1-1])
                                    grid_plot.set_ylim(0, im.shape[ax2-1])

                                    for row in xrange(self._pinning_matrix[ax1]):

                                        grid_plot.plot(
                                            np.ones(
                                            len(s_bfc)) * \
                                            s_bfr[row],
                                            np.array(s_bfc),
                                            'r-')

                                        for column in xrange(
                                                self._pinning_matrix[ax2]):

                                            grid_plot.plot(
                                                np.array(s_bfr),
                                                np.ones(
                                                len(s_bfr)) * \
                                                s_bfc[column],
                                                'r-')

                                    grid_plot.add_patch(plt.Rectangle((y2,x2),
                                        tf_im.shape[ax1-1],
                                        tf_im.shape[ax2-1],
                                        ls='solid', lw=2,
                                        fill=False, ec=(0.9, 0.9, .1, 1)))

                                    grid_image.savefig("gridding_error.png")

                                    err_str = "Image showing the grid that " +\
                                                "caused it: gridding_error.png"

                                    raise Exception(IndexError, err_str)

                                    sys.exit()

                    else:

                        self.logger.critical("ANALYSIS GRID ARRAY Lacks" + \
                                " transformation possibilities")

                    _cur_gc.set_data_source(tf_im)

                    #if watch_colony != None:
                        #if row == watch_colony[1] and column \
                                    #== watch_colony[2]:

                            #self.watch_scaled = tf_im
                            #if self.watch_scaled.sum() == \
                                    #(self.watch_scaled > 0).sum():

                                ###DEBUG WHAT IS THE GRID ARRAY
                                #plt.clf()
                                #plt.imshow(self.watch_scaled, title='Grid')
                                #plt.show()
                                ###END DEBUG CODE

                    #This happens only the first time
                    if self._first_analysis:

                        _cur_gc.attach_analysis(
                                blob=True, background=True, cell=True,
                                use_fallback_detection=use_fallback,
                                run_detect=False)

                    #This step only detects the objects
                    _cur_gc.get_analysis(no_analysis=True,
                                    remember_filter=True)

                    ###DEBUG RE-DETECT PART1
                    #debug_plt = plt.figure()
                    #debug_plt.add_subplot(221)
                    #plt.imshow(_cur_gc.get_item('blob').filter_array)
                    #debug_plt.add_subplot(223)
                    #plt.imshow(_cur_gc.get_item('blob').grid_array)
                    #plt.show()
                    #raw_input("> ")
                    ###DEBUG END PART1

                    #Transfer data to 'Cell Estimate Space'
                    bg_filter = _cur_gc.get_item('background').filter_array

                    if bg_filter.sum() == 0:

                        #debug_plt = plt.figure()
                        #debug_plt = plt.figure()
                        ##debug_plt.add_subplot(221)
                        #plt.imshow(_cur_gc.get_item('blob').filter_array)
                        #debug_plt.add_subplot(223)
                        #plt.imshow(_cur_gc.get_item('blob').grid_array)

                        #debug_plt.show()

                        #raw_input('x> ')

                        self.logger.warning('Time/Plate ' + \
                                '{0}, Row: {1}, Column: {2}'.format(
                                self._identifier, row, column) + \
                                ' has no background (skipping)')

                        self._features[row][col] = None

                    else:
                        _cur_gc.set_new_data_source_space(
                                space='cell estimate', bg_sub_source=bg_filter,
                                polynomial_coeffs=self._polynomial_coeffs)

                        #This step re-detects in Cell Estimate Space
                        #_cur_gc.get_analysis(
                            #no_analysis=True,\
                        #    remember_filter=True, use_fallback=True)

                        #analysis on the previously detected objects
                        self._features[row][col] = \
                                _cur_gc.get_analysis(no_detect=True)

                        ###DEBUG RE-DETECT PART2
                        #debug_plt.add_subplot(222)
                        #plt.imshow(_cur_gc.get_item('blob').filter_array)
                        #debug_plt.add_subplot(224)
                        #plt.imshow(_cur_gc.get_item('blob').grid_array)
                        #debug_plt.show()
                        #plot = raw_input('waiting: ')
                        ###DEBUG END

                    if watch_colony is not None \
                                and row == watch_colony[1] \
                                and column == watch_colony[2]:

                        blob = _cur_gc.get_item('blob')

                        background = _cur_gc.get_item('background')

                        if animate:
                            #plt.clf()
                            fig = plt.figure()
                            #gs = gridspec.GridSpec(2, 2)
                            #ax = fig.add_subplot(221, title="Blob")
                            #fig.gca().imshow(blob.filter_array)
                            #ax = fig.add_subplot(222, title ="Background")
                            #fig.gca().imshow(_cur_gc.get_item('background')\
                                #.filter_array)

                            #DEBUG CODE START
                            #blob = _cur_gc.get_item('blob')
                            #plt.clf()
                            #plt.subplot(211, title='filter all done')
                            #plt.imshow(blob.filter_array)
                            #plt.subplot(212, title='image')
                            #plt.imshow(blob.grid_array, vmax=3500, vmin=0)
                            #plt.show()
                            #DEBUG CODE END

                            ax = fig.add_subplot(221, title="Image t={0}".\
                                    format(self._identifier[0]))

                            ax_im = fig.gca().imshow(blob.grid_array, vmin=0,
                                    vmax=3500)

                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            #fig.colorbar(ax_im,ax)

                            ax = fig.add_subplot(223, title="Blob")
                            ax_im = fig.gca().imshow(blob.filter_array)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)

                            ax = fig.add_subplot(224, title="Background")
                            ax_im = fig.gca().imshow(background.filter_array)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)

                        if self._old_blob_img is not None and \
                                    self._old_timestamp is not None:

                            blob_diff = blob.get_diff(self._old_blob_img,
                                self._old_blob_filter)

                            #onion2 = blob.get_onion_values(blob_diff,
                                #self._old_blob_filter,
                                #2)
                            #onion2t = blob.get_onion_values(blob_diff,
                                #blob.filter_array,
                                #2)
                            #onion4 = blob.get_onion_values(blob_diff,
                                #self._old_blob_filter,
                                #4)

                            onion6 = blob.get_onion_values(blob_diff,
                                            self._old_blob_filter,
                                            6)

                            self._onion_store.insert(0,
                                    ((onion6[-1, 0] / onion6[-1, 1]) / \
                                    ((self._old_timestamp - timestamp) / \
                                    (3600.0))))

                            self._onion_times.insert(0, (timestamp + \
                                    (self._old_timestamp - timestamp) / 2.0) /\
                                    (3600.0))

                                #onion2t[-1,0]/float(onion2t[-1,1])))
                                #onion4[-1,0]/float(onion4[-1,1]),
                                #onion6[-1,0]/float(onion6[-1,1])))

                            #fig3 = plt.figure()
                            #fig3.add_subplot(2,2,1, title = 'now')
                            #fig3.gca().imshow(blob.grid_array, vmin=0,
                                    #vmax=3500)
                            #fig3.add_subplot(2,2,2, title = 'previous')
                            #fig3.gca().imshow(self._old_blob_img, vmin=0,
                                    #vmax=3500)
                            #fig3.add_subplot(2,2,3, title = 'previous')
                            #fig3.gca().imshow(im, vmin=0, vmax=3500)
                                #uncertain of if everything is cool here
                            #fig3.gca().plot([y2 - self._grid_cell_size[0]/2],
                                #[x2[0] - self._grid_cell_size[1]/2], 'ro')
                            #fig3.show()
                            #raw_input("now max {0} vs old {1}> ".format(
                                    #blob.grid_array.max(),
                                    #self._old_blob_img.max()))

                            if self._identifier[0] == 0:

                                onion_times = np.asarray(self._onion_times)
                                onion_store = np.asarray(self._onion_store)
                                onion_labels = ['T2 outer using true dt',
                                    'T2 outer using equal dt']
                                    #'Thickness 4, outer',
                                    #'Thickness 6, outer']

                                np.save('onion_start_val', np.array(
                                        (np.log2(onion6[-1, 0]),)))
                                np.save('onion_store_arr', onion_store)
                                np.save('onion_times_arr', onion_times)

                                #fig2 = plt.figure()
                                #fig2.gca().set_title("1st Derivative of "
                                   #"Outer Onion Peels (t vs t+1 onionrings")

                                #for i in xrange(onion_store.shape[0]):

                                    #fig2.gca().plot(
                                        #np.arange(onion_store.size),
                                        #np.arange(onion_store.shape[0]),
                                        #onion_store, '-',
                                        #label=onion_labels[0])

                                #fig2.gca().set_xlabel("Time indices")
                                #fig2.gca().set_ylabel("Avg cell estimate" +
                                    #" diff to next time-pt")
                                #fig2.gca().legend(loc=0)
                                #fig2.savefig("onion.png")

                            if animate:

                                ax = fig.add_subplot(222,
                                        title="Delta Cells Image")

                                ax_im = fig.gca().imshow(blob_diff,
                                        vmin=-700, vmax=700,
                                        cmap=plt.cm.RdYlGn)

                                ax.get_xaxis().set_visible(False)
                                ax.get_yaxis().set_visible(False)
                                fig.colorbar(ax_im)  # fraction=2)

                                #ax = fig.add_subplot(224,
                                        #title="Onion Avg Residuals")
                                #ax_im = fig.gca().plot(
                                        #np.arange(onion6.shape[0]),
                                        #onion6[:,0]/onion6[:,1]\
                                        #.astype(np.float64),
                                        #'g-')
                                #ax.set_xlabel(
                                    #'Onion layer index (0 = center of blob)')
                                #ax.set_ylabel('Avg residual(t+1 - t')
                                #ax.set_autoscalex_on(False)
                                #ax.set_autoscaley_on(False)
                                #ax.set_ylim((-150,300))
                                #ax.set_xlim((0,5))

                        else:

                            self._onion_times = []
                            self._onion_store = []

                        self._old_blob_img = blob.grid_array.copy()
                        self._old_blob_filter = blob.filter_array.copy()

                        if animate:
                            #ax = fig.add_subplot(313, title = "Growth-curve")
                            #fig.gca().semilogy(self.track_times,
                                #self.track_values,
                                #'b-', basey=2)
                            #self.track_times.append(self._identifier[0])
                            #self.track_values.append(
                                #self._features[row][column]\
                                #['blob']['pixelsum'])
                            #fig.gca().semilogy((self.track_times[-1],), (
                                #self.track_values[-1],),'ro', basey=2)
                            #ax.set_yticklabels(("0","2^5","1^6"))
                            #ax.set_yscale('log', basey=2)
                            #ax.set_yticks((0,5,6))
                            #plt.xlim(0, self.track_times[0])
                            #plt.ylim(0, max(self.track_values))
                            fig.savefig(save_anime_name)
                            del fig

                    if watch_colony != None:
                        if row == watch_colony[1] and \
                                    column == watch_colony[2]:

                            self.watch_blob = \
                                _cur_gc.get_item('blob').filter_array.copy()

                            self.watch_scaled = \
                                _cur_gc.get_item('blob').grid_array.copy()

                            self.watch_results = self._features[row][column]

        self._old_timestamp = timestamp

        #print str(((row+1)*self._pinning_matrix[1]+column+1)/total_steps)+"%"
        self._first_analysis = False

        return self._features
