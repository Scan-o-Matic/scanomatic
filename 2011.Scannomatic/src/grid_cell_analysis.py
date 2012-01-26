#! /usr/bin/env python


#
# DEPENDENCIES
#

import cv
import numpy as np
import numpy.ma as ma
import math
from scipy.stats.mstats import mquantiles, tmean, trim

#
# SCANNOMATIC LIBRARIES
#

import histogram as hist

#
# FUNCTIONS
#

def points_in_circle(circle, arr):
    """
        A generator to return all points whose indices are within given circle.

        Function takes two arguments:

        @circle     A tuple with the structure ((i,j),r)
                    Where i and j are the center coordinates of the arrays first
                    and second dimension

        @arr        An array

        Usage:

        raster = np.fromfunction(lambda i,j: 100+10*i+j, shape, dtype=int)
        points_iterator = points_in_circle((i0,j0,r),raster)
        pts = np.array(list(points_iterator))

        Originally written by jetxee
        Modified by Martin Zackrisson

        Found on 
            http://stackoverflow.com/questions/2770356/extract-points-within-a-shape-from-a-raster
    """
    origo, r = circle
    def intceil(x):
        return int(np.ceil(x))

    for i in xrange(intceil(origo[0]-r),intceil(origo[0]+r)):
        ri = np.sqrt(r**2-(i-i0)**2)
        for j in xrange(intceil(origo[1]-ri),intceil(origo[1]+ri)):
            yield arr[i][j]

    
#
# CLASSES Cell_Item
#

class Cell_Item():

    def __init__(self):
        """
            Cell_Item is a super-class for Blob, Backgroun and Cell and should
            not be accessed directly.

            It has two functions:

            set_type    checks and defines the type of cell item a thing is

            do_analysis     runs analysis on a cell type, given that it has
                            previously been detected

        """

        self.features = {}
        self.CELLITEM_TYPE = 0
        self.set_type()

    #
    # SET functions
    #

    def set_data_source(self, data_source):
        self.grid_array = data_source

    def set_type(self):

        """
            This function empties the features-dictionary (as a precausion)
            and sets the cell item type.

            The function takes no arguments

        """

        self.features = {}

        if isinstance(self, Blob):
            self.CELLITEM_TYPE = 1
        elif isinstance(self, Background):
            self.CELLITEM_TYPE = 2
        elif isinstance(self, Cell):
            self.CELLITEM_TYPE = 3

    #
    # DO functions
    #

    def do_analysis(self):

        """
            do_analysis updates the values of the features-dict.
            Depending one what type of cell item it is (Blob, Background, Cell)
            different types of calculations will be done.

            The function requires that the cell item type has been set,
            which can be ensured by running set_type.

            Default initiation of a cell item will automatically set the type.

            The function takes no arguments

        """
        if self.CELLITEM_TYPE == 0 or self.filter_array == None:
            return None


        self.features['area'] = self.filter_array.sum()
        self.features['pixelsum'] = np.float64((self.grid_array * self.filter_array).sum())


        if self.CELLITEM_TYPE == 1:
            self.features['centroid'] = None
            self.features['perimeter'] = None

        if self.CELLITEM_TYPE == 3:
            self.features['median'] = np.median(self.grid_array)
            self.features['mean'] = self.grid_array.mean()
            self.features['IQR'] = mquantiles(self.grid_array,prob=[0.25,0.75])
            self.features['IQR_mean'] = tmean(self.grid_array,self.features['IQR'])
        else:
            feature_array = ma.masked_array(self.grid_array, mask=abs(self.filter_array - 1))
            self.features['median'] = ma.median(feature_array)
            self.features['mean'] = feature_array.mean()
            self.features['IQR'] = mquantiles(ma.compressed(feature_array),prob=[0.25,0.75])
            try:
                self.features['IQR_mean'] = ma.masked_outside(feature_array, self.features['IQR'][0], self.features['IQR'][1]).mean()
            except:
                self.features['IQR_mean'] = None
                print "*** Failed in producting IQR_mean from IQR", self.features['IQR']
#        if self.CELLITEM_TYPE == 3: # or self.CELLITEM_TYPE == 1:
#            #Using IQR as default colony_size measure
#            self.features['colony_size'] = self.features['pixelsum'] - \
#                self.features['IQR_mean'] * self.features['area']
#            self.features['colony_size_from_mean'] = self.features['pixelsum'] - \
#                self.features['mean'] * self.features['area']
#            self.features['colony_size_from_median'] = self.features['pixelsum'] - \
#                self.features['median'] * self.features['area']

#
# CLASS Blob
#

class Blob(Cell_Item):
    def __init__(self, grid_array, run_detect=True, threshold=None, use_fallback_detection=False, 
        image_color_logic = "Norm"):

        Cell_Item.__init__(self)

        self.grid_array = grid_array
        self.threshold = threshold
        self.use_fallback_detection = use_fallback_detection
        self.filter_array = None
        self.image_color_logic = image_color_logic

        self.histogram = hist.Histogram(self.grid_array, run_at_init = False)        

        if run_detect:
            if self.use_fallback_detection == True:
                self.threshold_detect()
            else:
                self.edge_detect()
    #
    # SET functions
    #

    def set_blob_from_shape(self, rect=None, circle=None):
        """
            set_blob_from_shape serves as the purpose of allowing users to
            define their blob (that is where the colony is).

            It can take either a rectange or a circle description

            Arguments:

            @rect   A list of two two tuples.
                    First tuple should be that (upper, left) coordinate
                    Second tuple should be the (lower, right) coordinate

            @circle A tuple containing (origo, radius)
                    Where origo is a tuple itself (x,y)
        """

        if rect:
            self.filter_array = np.zeros(self.grid_array.shape, dtype=bool)
            self.filter_array[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]] = True

        elif circle:
            raster = np.fromfunction(lambda i,j: 100+10*i+j, self.grid_array.shape, dtype=int)
            pts_iterator = points_in_circle(circle, raster)
            self.filter_array = np.array(list(pts_iterator), dtype=bool)


    def set_threshold(self, threshold=None, relative=False):
        """

            set_threshold allows user to set the threshold manually or, if no
            argument is passed, to have it set using the histogram of the
            image section and the Otsu-algorithm

            Function has optional arguments

            @threshold      Manually enforced threshold
                            Default (None)

            @relative       Boolean declaring if threshold is a relative value.
                            This argument only has an effect togeather with 
                            threshold.
                            Default (false)
        """

        if threshold:
            if relative:
                self.threshold += threshold
            else:
                self.threshold = threshold
        else:

            self.threshold = hist.otsu(
                self.histogram.re_hist(self.grid_array))
            
    #
    # DETECT functions
    #

    def detect(self, use_fallback_detection = None):
        """
            Generic wrapper function for blob-detection

            Optional argument:

            @use_fallback_detection     If set, overrides the instance default        
        """

        if use_fallback_detection:
            self.threshold_detect()
        elif use_fallback_detection == False:
            self.edge_detect()
        elif self.use_fallback_detection:
            self.threshold_detect()
        else:
            self.edge_detect()

    def threshold_detect(self):
        """
            If there is a threshold previously set, this will be used to
            detect blob by accepting everythin above threshold as the blob.

            If no threshold has been set, threshold is calculated using
            Otsu on the histogram of the image-section.

            Function takes no arguments

        """

        if self.threshold == None:
            self.set_threshold()

        if self.image_color_logic == "inv":
            self.filter_array = (self.grid_array > self.threshold)
        else:
            self.filter_array = (self.grid_array < self.threshold)
         

    def edge_detect(self):
        """
            Edge detect actually includes the fallback threshold detect
            as first step.

            The function convolves a circular pattern over the thresholded
            image using first an erode function to discard noise and then
            a dilate function to expand the detected blobs.

            Finally the contours are aquired and the minimum circle is fitted
            ontop. Whereupon detection is discarded if the detected area is
            too small.

            Function takes no arguments.

        """

        #Threshold the image
        self.threshold_detect()

        #Not neccesary per se, but we need a copy anyways
        mat = cv.fromarray(self.filter_array)

        #Erode, radius 6, iterations = default (1)
        radius = 6
        kernel = cv.CreateStructuringElementEx(radius*2+1, radius*2+1,
            radius, radius, cv.CV_SHAPE_ELLIPSE)

        print "Kernel in place", kernel
        cv.Erode(mat, mat, kernel)
        print "Eroded"
        #Dilate, radius 4, iterations = default (1)
        radius = 4
        kernel = cv.CreateStructuringElementEx(radius*2+1, radius*2+1,
            radius, radius, cv.CV_SHAPE_ELLIPSE)

        print "Kernel in place"
        cv.Dilate(mat, mat, kernel)
        print "Dilated"

        #Bwareopen, min_radius 
        min_radius = 10
        contour = cv.FindContours(mat, cv.CreateMemStorage(), cv.CV_RETR_LIST)
        print "Found contours"
        self.bounding_circle = cv.MinEnclosingCircle(list(contour)) 
        if type(circle) == types.IntType or circle[2] < min_radius:
            print self.bounding_circle, "Too small or strange"
        else:
            print self.bountin_circle, "is good"
            self.filter_array = numpy.asarray(mat)            
            
#
# CLASSES Background (invers blob area)
#

class Background(Cell_Item):
    def __init__(self, grid_array, blob, run_detect=True):

        Cell_Item.__init__(self)

        self.grid_array = grid_array
        if isinstance(blob, Blob):
            self.blob = blob
        else:
            self.blob = None

        if run_detect:
            self.detect()

    def detect(self):
        """

            detect finds the background

            It is assumed that the background is the inverse
            of the blob. Therefore this function only runs after
            the detect function has been run on blob.

            Function takes no arguments

        """

        if self.blob and self.blob.filter_array != None:
            self.filter_array = (self.blob.filter_array == False)

#
# CLASSES Cell (entire area)
#

class Cell(Cell_Item):
    def __init__(self, grid_array, run_detect=True, threshold=-1):

        Cell_Item.__init__(self)

        self.grid_array = grid_array
        self.threshold = threshold

        self.filter_array = None

        if run_detect:
            self.detect()

    def detect(self):
        """

            detect makes a filter that is true for the full area

            The function takes no argument.

        """

        self.filter_array = np.ones(self.grid_array.shape, dtype=bool)
        
