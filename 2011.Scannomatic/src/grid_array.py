#! /usr/bin/env python

# 
# colonies.py   v 0.1
#
# This is a convienience module for command line calling of all different types of colony
# analysis that are implemented.
#
# The module can also be imported directly into other scrips as a wrapper
#



#
# DEPENDENCIES
#


#
# SCANNOMATIC LIBRARIES
#

import grid_array_analysis as gaa
import grid_cell

#
# CLASS: Grid_Array
#

class Grid_Array():
    def __init__(self, pinning_matrix):

        self._analysis = gaa.Grid_Analysis()
        self._pinning_matrix = None
        self._grid_cell_size = None
        self._grid_cells = None
        self._features = []

        self.R = None
        if pinning_matrix != None:
            self.set_pinning_matrix(pinning_matrix)
    #
    # SET functions
    #

    def set_pinning_matrix(self, pinning_matrix):
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
                self._grid_cells[row].append(grid_cell.Grid_Cell())
                self._features[row].append(None)

    #
    # Get functions
    # 

    def get_analysis(self, im, use_fallback=False, use_otsu=True, median_coeff=None, verboise=False, visual=False):

        """
            @im         An array / the image

            @use_otsu   Causes thresholding to be done by Otsu
                        algorithm (Default)

            @median_coeff       Coefficient to threshold from the
                                median when not using Otsu.

            @verboise   If a lot of things should be printed out

            @visual     If visual information should be presented.

            The function returns two arrays, one per dimension, of the
            positions of the spikes and a quality index

        """

        #DEBUGHACK
        #visual = True
        #verboise = True
        #DEBUGHACK - END

        best_fit_rows, best_fit_columns, R = self._analysis.get_analysis(im, self._pinning_matrix, use_otsu, median_coeff, verboise, visual)

        self.R = R

        if verboise:
            print "*** Grid (rows x columns):"
            print best_fit_rows
            print best_fit_columns
            print

        if best_fit_rows == None or best_fit_columns == None:
            return None

        rect_size = None

        has_previous_rect = True

        if self._grid_cell_size == None:
            self._grid_cell_size = self._analysis.best_fit_frequency
            rect_size = self._grid_cell_size
            has_previous_rect = False
 
        total_steps = float(self._pinning_matrix[0] * self._pinning_matrix[1])
        #print "*** Analysing grid:"
        #import matplotlib.pyplot as plt
        #plt.imshow(im)

        for row in xrange(self._pinning_matrix[0]):
            for column in xrange(self._pinning_matrix[1]):

                self._grid_cells[row][column].set_center( \
                    (best_fit_rows[row], best_fit_columns[column]) , rect_size)

                ul = self._grid_cells[row][column].get_top_left()
                lr = self._grid_cells[row][column].get_bottom_right()
                self._grid_cells[row][column].set_data_source( im[ul[1]:lr[1],ul[0]:lr[0]] )

                #plt.plot(self._grid_cells[row][column].center[0],
                    #self._grid_cells[row][column].center[1] , 'k.')

                               
                #This happens only the first time
                if has_previous_rect == False:
                    self._grid_cells[row][column].attach_analysis(
                        blob=True, background=True, cell=True, 
                        use_fallback_detection=use_fallback, run_detect=False)


                self._features[row][column] = \
                    self._grid_cells[row][column].get_analysis()

        #plt.show()
                #print str(((row+1)*self._pinning_matrix[1] + column+1)/total_steps) + "%"
        return self._features 
