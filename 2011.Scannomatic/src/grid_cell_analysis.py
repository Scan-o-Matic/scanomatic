#! /usr/bin/env python


#
# DEPENDENCIES
#

#import cv
import numpy as np
import numpy.ma as ma
import math
from scipy.stats.mstats import mquantiles, tmean, trim
from scipy.ndimage.filters import sobel 
from scipy.ndimage import binary_erosion, binary_dilation,\
    binary_fill_holes, binary_closing, center_of_mass, label, laplace,\
    gaussian_filter
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
            try:
                self.features['IQR_mean'] = tmean(self.grid_array,self.features['IQR'])
            except:
                self.features['IQR_mean'] = None

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
    def __init__(self, grid_array, run_detect=True, threshold=None, \
        use_fallback_detection=False, image_color_logic = "Norm", \
        center=None, radius=None):

        Cell_Item.__init__(self)

        self.grid_array = grid_array
        self.threshold = threshold
        self.use_fallback_detection = use_fallback_detection
        self.filter_array = None
        self.image_color_logic = image_color_logic

        self.histogram = hist.Histogram(self.grid_array, run_at_init = False)        

        if run_detect:
            if center is not None and radius is not None:
                self.manual_detect(center, radius)
            elif self.use_fallback_detection == True:
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


    def set_threshold(self, threshold=None, relative=False, im=None):
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

            @im             Optional alternative image source

        """

        if threshold:
            if relative:
                self.threshold += threshold
            else:
                self.threshold = threshold
        else:

            if im is None:
                im = self.grid_array

            self.threshold = hist.otsu(
                self.histogram.re_hist(im))
            
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

    def threshold_detect(self, im=None, threshold=None, color_logic=None):
        """
            If there is a threshold previously set, this will be used to
            detect blob by accepting everythin above threshold as the blob.

            If no threshold has been set, threshold is calculated using
            Otsu on the histogram of the image-section.

            Function takes one optional argument:

            @im             Optional alternative image source

        """

        if self.threshold == None:
            self.set_threshold(im=im, threshold=threshold)

        if im is None:
            im = self.grid_array        

        if color_logic is None:
            color_logic = self.image_color_logic

        if color_logic == "inv":
            self.filter_array = (im > self.threshold)
        else:
            self.filter_array = (im < self.threshold)
        
    def manual_detect(self, center, radius):

        self.filter_array = np.zeros(self.grid_array.shape)

        stencil = self.get_round_kernel(int(np.round(radius)))
        x_size = (stencil.shape[0] - 1) / 2
        y_size = (stencil.shape[1] - 1) / 2

        if stencil.shape == \
            self.filter_array[center[0] - x_size : center[0] + x_size +1,\
            center[1] - y_size : center[1] + y_size + 1].shape:
        
            self.filter_array[center[0] - x_size : center[0] + x_size + 1, 
                center[1] - y_size : center[1] + y_size + 1] += stencil

        else:

            self.edge_detect()
 
    def get_round_kernel(self, radius=6, outline=False):


        round_kernel = np.zeros(((radius+1)*2+1,(radius+1)*2+1)).astype(\
            self.filter_array.dtype)

        center_x, center_y = radius+1, radius+1

        y,x = np.ogrid[-radius:radius, -radius:radius]

        if outline:
            index = radius**2 - 1 <= x**2 + y**2 <= radius** + 2
        else:
            index = x**2 + y**2 <= radius**2

        round_kernel[center_y-radius:center_y+radius, 
            center_x-radius:center_x+radius][index] = True

        return round_kernel

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

        self.old_edge_detect()

    def edge_detect_sobel(self):

        from matplotlib import pyplot

        #De-noising the image with a smooth
        self.filter_array = gaussian_filter(self.grid_array, 2)
        pyplot.imshow(self.filter_array)
        pyplot.savefig('blob_gauss.png')
        pyplot.clf()        
         
        #Checking the second dirivative
        #self.filter_array = laplace(self.filter_array)
        self.filter_array = sobel(self.filter_array)

        pyplot.imshow(self.filter_array)
        #pyplot.savefig('blob_laplace.png')
        pyplot.savefig('blob_sobel.png')
        pyplot.clf()        

        #self.filter_array = gaussian_filter(self.filter_array, 2)
        #pyplot.imshow(self.filter_array)
        #pyplot.savefig('blob_gauss2.png')
        #pyplot.clf()        

        #Thesholding the edges
        self.threshold_detect(im=self.filter_array, color_logic = 'inv',
            threshold=np.max(self.filter_array)*0.2)


        pyplot.imshow(self.filter_array)
        pyplot.savefig('blob_theshold.png')
        pyplot.clf()        


        kernel = self.get_round_kernel(radius=2)
        self.filter_array = binary_erosion(self.filter_array, structure=kernel)

        pyplot.imshow(self.filter_array)
        pyplot.savefig('blob_erosion.png')
        pyplot.clf()        

        kernel = self.get_round_kernel(radius=3)
        self.filter_array = binary_dilation(self.filter_array, structure=kernel)

        pyplot.imshow(self.filter_array)
        pyplot.savefig('blob_dilation.png')
        pyplot.clf()        

        label_array, number_of_labels = label(self.filter_array)
        #print number_of_labels
        kernel = self.get_round_kernel(radius=1)
        center_size = 2
        circle_parts = []

        for i in xrange(number_of_labels):
            cur_item = (label_array == (i+1))
            cur_pxs = np.sum(cur_item)
            if cur_pxs > 100:
                #c_o_m = center_of_mass(cur_item)
                #print "Mass centra: ", c_o_m        
                oneD = np.where(np.sum(cur_item,1) > 0 )[0]
                dim1 = (oneD[0], oneD[-1])
                oneD = np.where(np.sum(cur_item,0) > 0 )[0]
                dim2 = (oneD[0], oneD[-1])
                cur_off = 2
                good_part = True

                if cur_item[dim1[0]:dim1[1], dim2[0]+cur_off].sum() / \
                    float(cur_pxs)\
                    >= cur_pxs / float(dim2[1] - dim2[0]):
            
                    good_part = False

                if good_part and cur_item[dim1[0]:dim1[1], \
                    dim2[1]-cur_off].sum() / float(cur_pxs)\
                    >= cur_pxs / float(dim2[1] - dim2[0]):
            
                    good_part = False

                if cur_item[dim2[0]:dim2[1], dim1[0]+cur_off].sum() / \
                    float(cur_pxs)\
                    >= cur_pxs / float(dim1[1] - dim1[0]):
            
                    good_part = False

                if good_part and cur_item[dim2[0]:dim2[1], \
                    dim1[1]-cur_off].sum() / float(cur_pxs)\
                    >= cur_pxs / float(dim1[1] - dim1[0]):
            
                    good_part = False


                #if cur_item[c_o_m[0]-center_size:c_o_m[0]+center_size,
                #    c_o_m[1]-center_size: c_o_m[1]+center_size].sum() > 0:

                if good_part == False:
                    pyplot.imshow(binary_erosion((label_array == (i+1)),
                        structure=kernel))
                    pyplot.savefig('blob_item_' + str(i+1) + '_bad.png')
                    pyplot.clf()        
                    #print np.sum((label_array == (i+1)))
                else:
                    pyplot.imshow(binary_erosion((label_array == (i+1)),
                        structure=kernel))
                    pyplot.savefig('blob_item_' + str(i+1) + '.png')
                    pyplot.clf()        
                    #print np.sum((label_array == (i+1)))
                    circle_parts.append(i+1)

        self.filter_array = np.zeros(self.filter_array.shape)

        for c_part in circle_parts:
            #print self.filter_array.shape, label_array.shape
            self.filter_array += (label_array == c_part)



    def old_edge_detect(self):

        #De-noising the image with a smooth
        self.filter_array = gaussian_filter(self.grid_array, 2)

        #Threshold the image
        self.threshold_detect(im=self.filter_array)

        #self.filter_array = sobel(self.grid_array)

        #print np.sum(self.filter_array), "pixels inside at this stage"
        #from scipy.ndimage.filters import sobel 
        #from scipy.ndimage.morpholgy import binary_erosion, binary_dilation,
        #    binary_fill_holes, binary_closing

        #Not neccesary per se, but we need a copy anyways
        #mat = cv.fromarray(self.filter_array)
        #print "**Mat made"
        #eroded_mat = cv.CreateMat(mat.rows, mat.cols, cv.CV_8UC1)

        #Erosion kernel
        kernel = self.get_round_kernel(radius=3)
        #print kernel.astype(int)
        #print "***Erosion kernel ready"

        self.filter_array = binary_erosion(self.filter_array, structure=kernel)

        #Erode, radius 6, iterations = default (1)
        #kernel = cv.CreateStructuringElementEx(radius*2+1, radius*2+1,
        #    radius, radius, cv.CV_SHAPE_ELLIPSE)

        #print "Kernel in place", kernel
        #cv.Erode(mat, eroded_mat, kernel)
        #print "Eroded"
        #print np.sum(self.filter_array), "pixels inside at this stage"
        #Dilate, radius 4, iterations = default (1)
        #radius = 4
        #kernel = cv.CreateStructuringElementEx(radius*2+1, radius*2+1,
        #    radius, radius, cv.CV_SHAPE_ELLIPSE)

        kernel = self.get_round_kernel(radius=4)

        #print "Kernel in place"
        self.filter_array = binary_dilation(self.filter_array, structure=kernel)
        #cv.Dilate(mat, mat, kernel)
        #print "Dilated"
        #print np.sum(self.filter_array), "pixels inside at this stage"

        self.filter_array = binary_closing(self.filter_array, structure=kernel)

        #print "Closing applied"
        #print np.sum(self.filter_array), "pixels inside at this stage"


        #self.filter_array = sobel(self.filter_array)

        #print "Edged detected"
        #print np.sum(self.filter_array), "pixels inside at this stage"

        self.filter_array = binary_fill_holes(self.filter_array, 
            structure=np.ones((5,5)))

        #print "Holes filled"
        #print np.sum(self.filter_array), "pixels inside at this stage"


        label_array, number_of_labels = label(self.filter_array)
        qualities = []

        if number_of_labels > 1:

            for item in xrange(number_of_labels):

                cur_item = (label_array == (item + 1))
                cur_pxs = np.sum( cur_item ) 

                oneD = np.where(np.sum(cur_item,1) > 0 )[0]
                dim1 = oneD[-1] -  oneD[0]
                oneD = np.where(np.sum(cur_item,0) > 0 )[0]
                dim2 = oneD[-1] -  oneD[0]

                if dim1 > dim2:

                    qualities.append(cur_pxs * dim2 / float(dim1))
                
                else:

                    qualities.append(cur_pxs * dim1 / float(dim2))

            q_best = np.asarray(qualities).argmax()

            self.filter_array = (label_array == (q_best + 1) )

        #axis_1 = np.mean(self.filter_array,1)
        #axis_2 = np.mean(self.filter_array,0)

        #ax_1_center = np.argmax(axis_1)
        #ax_2_center = np.argmax(axis_2)

        #borders = np.where(axis_1 == 0)

        #Bwareopen, min_radius 
        #min_radius = 10
        #contour = cv.FindContours(mat, cv.CreateMemStorage(), cv.CV_RETR_LIST)
        #print "Found contours"
        #self.bounding_circle = cv.MinEnclosingCircle(list(contour)) 
        #if type(circle) == types.IntType or circle[2] < min_radius:
        #    print self.bounding_circle, "Too small or strange"
        #else:
        #    print self.bountin_circle, "is good"
        #    self.filter_array = numpy.asarray(mat)            
            
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
        
