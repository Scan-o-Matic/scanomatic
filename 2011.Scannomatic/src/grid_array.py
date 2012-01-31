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

import numpy as np

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

    def get_transformation_matrix(self, gs_values=None, gs_fit=None, y_range = (0,255)):
        """
            get_transformation_matrix takes an coefficient array of a
            polynomial fit of the 3rd degree and calculates a matrix
            of all solutions for all the integer steps of the y-range
            specified.

            The function takes two arguments:

            @gs_values  A numpy array or a list of gray-scale values

            @gs_fit     A numpy array of the coefficients as returned
                        by numpy.polyfit, assuming 3rd degree 
                        solution
    
            @y_range    A tuple having the including range limits 
                        for the solution.

            The function returns a list of transformation values

        """

        if gs_values != None:

            gs_indices  = range(len(gs_values))

            tf_matrix = np.zeros((y_range[1]+1))

            for pos in xrange(len(gs_values)):

                #Ofsetting local box at edges
                if pos == 0:
                    offset = 1
                elif pos == len(gs_values)-1:
                    offset = -1
                else:
                    offset = 0

                #create 1st degree polynomial
                p = np.poly1d(np.polyfit(gs_indices[ pos + offset - 1 : pos + offset + 2 ],\
                    gs_values[ pos + offset - 1: pos + offset + 2], 1))

                a = p.c[0]
                b = p.c[1]

                #Getting formula ax + b = 0
                #b = np.interp(0,gs_indices[ pos + offset - 1 : pos + offset + 1 ],\
                    #gs_values[ pos + offset - 1: pos + offset + 1])

                #Using x as 1
                #a = (np.interp(y_range[1]+1.0, gs_indices[ pos + offset - 1: pos + offset + 1],\
                    #gs_values[ pos + offset -1 : pos + offset + 1]) - b) / (y_range[1]+1.0)

                #Setting limits to the local interpolation
                y_left = p(gs_indices[ pos + offset - 1 ])
                #y_left = np.interp(gs_indices[ pos + offset - 1] ,\
                    #gs_indices[ pos + offset - 1: pos + offset + 1],\
                    #gs_values[ pos + offset -1 : pos + offset + 1])

                y_right= p(gs_indices[ pos + offset ])
                #y_right = np.interp(gs_indices[ pos + offset ],\
                    #gs_indices[ pos + offset - 1: pos + offset + 1],\
                    #gs_values[ pos + offset - 1 : pos + offset + 1])

                #Orderings bounds accending
                if y_left > y_right:
                    tmp = y_left
                    y_left = y_right
                    y_right = tmp

                #Extending at edges
                if pos == 0 or pos == len(gs_values) -1:
                    if abs(y_left - y_range[0]) < abs(y_right - y_range[1]):
                        y_left = y_range[0]
                    else:
                        y_right = y_range[1]
         
                #Trimming so inside y_range
                if y_left < y_range[0]:
                    y_left = y_range[0]
                if y_right > y_range[1]:
                    y_right = y_range[1]

                #Inserting values for the tf_matrix
                for y_pos in range(int(y_left), int(np.ceil(y_right))+1):
                    tf_matrix[y_pos] = (y_pos - b) / a

        else:    

            tf_matrix = []

            for y in range(y_range[0],y_range[1]+1):

                #Do something real here
                #The caluclated value shoud be a float
                x = y
                
                tf_matrix.append(x)

        return tf_matrix

    def get_analysis(self, im, gs_fit=None, gs_values=None, use_fallback=False,\
        use_otsu=True, median_coeff=None, verboise=False, visual=False, watch_colony=None, supress_other=False):

        """
            @im         An array / the image

            @gs_fit     An array of the fitted coefficients for the grayscale

            @gs_values  An array of the grayscale pixelvalues, if submittet 
                        gs_fit is disregarded

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

        self.watch_source = None
        self.watch_scaled = None
        self.watch_blob = None
        self.watch_results = None

        best_fit_rows, best_fit_columns, R = self._analysis.get_analysis(im, self._pinning_matrix, use_otsu, median_coeff, verboise, False) #visual #instead of false there...

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
            #print self._grid_cell_size
            rect_size = self._grid_cell_size
            has_previous_rect = False
 
        #total_steps = float(self._pinning_matrix[0] * self._pinning_matrix[1])


        #Normalising towards grayscale before anything is done on the colonies
        transformation_matrix = None
        if gs_values == None:
            transformation_matrix = self.get_transformation_matrix(gs_fit=gs_fit)
        else:
            transformation_matrix = self.get_transformation_matrix(gs_values = gs_values)

        #if watch_colony != None:
        #    ul = self._grid_cells[watch_colony[1]][watch_colony[2]].get_top_left()
        #    lr = self._grid_cells[watch_colony[1]][watch_colony[2]].get_bottom_right()
        #    self.watch_source = im[ul[1]:lr[1],ul[0]:lr[0]]


        #if transformation_matrix != None:
            #There's probably some faster way
            #for x in xrange(im.shape[0]):
                #for y in xrange(im.shape[1]):
                    #im[x,y] = transformation_matrix[im[x,y]]
        #print "*** Analysing grid:"
        if visual:
            import matplotlib.pyplot as plt
            plt.imshow(im)

        for row in xrange(self._pinning_matrix[0]):
            if visual:
                plt.plot(\
                    np.ones(len(best_fit_columns))*best_fit_rows[row],
                    np.array(best_fit_columns),
                    'r-')
            for column in xrange(self._pinning_matrix[1]):
                if visual:
                    plt.plot(\
                        np.array(best_fit_rows),
                        np.ones(len(best_fit_rows))*best_fit_columns[column],
                        'r-')
                if supress_other == False or (watch_colony != None and \
                        watch_colony[1] == row and watch_colony[2] == column):

                    self._grid_cells[row][column].set_center( \
                        (best_fit_rows[row], best_fit_columns[column]) , rect_size)

                    coord_2nd = self._grid_cells[row][column].get_first_dim_as_tuple()
                    coord_1st = self._grid_cells[row][column].get_second_dim_as_tuple()
                    #ul = self._grid_cells[row][column].get_top_left()
                    #lr = self._grid_cells[row][column].get_bottom_right()
                    if transformation_matrix != None:
                        #There's probably some faster way
                        #print coord_1st, coord_2nd, im.shape
                        for x in xrange(int(coord_1st[0]),int(np.ceil(coord_1st[1]))):
                            for y in xrange(int(coord_2nd[0]),int(np.ceil(coord_2nd[1]))):
                                im[x,y] = transformation_matrix[im[x,y]]

                    self._grid_cells[row][column].set_data_source( \
                        im[coord_1st[0]:coord_1st[1],\
                        coord_2nd[0]:coord_2nd[1]] )


                    if watch_colony != None:
                        if row == watch_colony[1] and column == watch_colony[2]:
                            self.watch_scaled = im[coord_1st[0]:coord_1st[1],\
                                coord_2nd[0]:coord_2nd[1]]
                            
                    if visual:
                        pass
                        #plt.plot(self._grid_cells[row][column].center[0],
                        #    self._grid_cells[row][column].center[1] , 'k.')

                        #plt.plot(np.asarray(coord_1st),np.asarray(coord_2nd),'k')              

                    #This happens only the first time
                    if has_previous_rect == False:
                        self._grid_cells[row][column].attach_analysis(
                            blob=True, background=True, cell=True, 
                            use_fallback_detection=use_fallback, run_detect=False)


                    self._features[row][column] = \
                        self._grid_cells[row][column].get_analysis()

                    if watch_colony != None:
                        if row == watch_colony[1] and column == watch_colony[2]:
                            self.watch_blob = self._grid_cells[row][column]._analysis_items['blob'].filter_array
                            self.watch_results = self._features[row][column]
        if visual:    
            plt.show()
                #print str(((row+1)*self._pinning_matrix[1] + column+1)/total_steps) + "%"
        return self._features 
