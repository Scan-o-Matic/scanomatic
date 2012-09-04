#!/usr/bin/env python
"""
Part of the analysis work-flow that analyses the image section of a grid-cell.
"""
__author__ = "Martin Zackrisson, jetxee"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Mats Kvarnstroem", "Andreas Skyman",
    "jetxee"]
__license__ = "GPL v3.0"
__version__ = "0.996"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

#import cv
import numpy as np
import math
#import logging
from scipy.stats.mstats import mquantiles, tmean, trim
from scipy.ndimage.filters import sobel
from scipy.ndimage import binary_erosion, binary_dilation,\
    binary_fill_holes, binary_closing, center_of_mass, label, laplace,\
    gaussian_filter, median_filter
#
# SCANNOMATIC LIBRARIES
#

import resource_histogram as hist
import resource_logger as logger
import resource_blob as rblob

#
# FUNCTIONS
#


def points_in_circle(circle):
    """A generator to return all points whose indices are within given circle.

    Function takes two arguments:

    @circle     A tuple with the structure ((i,j),r)
                Where i and j are the center coordinates of the arrays first
                and second dimension

    @arr        An array (NOT ANYMORE, just give positions!)

    Usage:

    raster = np.fromfunction(lambda i,j: 100+10*i+j, shape, dtype=int)
    points_iterator = points_in_circle(((i0,j0),r),raster)
    pts = np.array(list(points_iterator))

    Originally written by jetxee
    Modified by Martin Zackrisson

    Found on
    http://stackoverflow.com/questions/2770356/
    extract-points-within-a-shape-from-a-raster
    """

    (i0, j0), r = circle

    def intceil(x):
        return int(np.ceil(x))

    for i in xrange(intceil(i0 - r), intceil(i0 + r)):

        ri = np.sqrt(r ** 2 - (i - i0) ** 2)

        for j in xrange(intceil(j0 - ri), intceil(j0 + ri)):

            yield (i, j)
            #yield arr[i][j]


def get_round_kernel(radius=6, outline=False):

    round_kernel = np.zeros(((radius + 1) * 2 + 1,
                    (radius + 1) * 2 + 1))

    center_1D = radius + 1

    y, x = np.ogrid[-radius: radius, -radius: radius]

    if outline:

        index = radius ** 2 - 1 <= x ** 2 + y ** 2 <= radius ** 2 + 2

    else:

        index = x ** 2 + y ** 2 <= radius ** 2

    round_kernel[center_1D - radius: center_1D + radius,
            center_1D - radius: center_1D + radius][index] = True

    return round_kernel


def get_round_kernel2(radius=3):

    round_kernel = np.zeros(((radius + 1) * 2 + 1,
                    (radius + 1) * 2 + 1))

    center_1D = radius + 1

    circle = points_in_circle(((center_1D, center_1D), radius))

    for pt in circle:

        round_kernel[pt] = 1

    return round_kernel


def get_array_subtraction(A1, A2, offset, output=None):
    """Makes offsetted subtractions for A1 - A2 independent of sizes

    If output is supplied it will be fed directly into it, else,
    it will just return a new array.
    """

    o1_low = offset[0]
    o2_low = offset[1]

    o1_high = o1_low + A2.shape[0]
    o2_high = o2_low + A2.shape[1]

    if o1_low < 0:

        b1_low = -o1_low
        o1_low = 0

    else:

        b1_low = 0

    if o2_low < 0:

        b2_low = -o2_low
        o2_low = 0

    else:

        b2_low = 0

    if o1_high > A1.shape[0]:

        b1_high = A2.shape[0] - (o1_high - A1.shape[0])
        o1_high = A1.shape[0]

    else:

        b1_high = A2.shape[0]

    if o2_high > A1.shape[1]:

        b2_high = A2.shape[1] - (o2_high - A1.shape[1])
        o2_high = A1.shape[1]

    else:

        b2_high = A2.shape[1]

    if output is None:

        diff_array = A1.copy()

        diff_array[o1_low: o1_high, o2_low: o2_high] -= \
                    A2[b1_low: b1_high, b2_low: b2_high]

        return diff_array

    else:

        output[o1_low: o1_high, o2_low: o2_high] = \
                    A1[o1_low: o1_high, o2_low: o2_high] - \
                    A2[b1_low: b1_high, b2_low: b2_high]

#
# CLASSES Cell_Item
#


class Cell_Item():

    def __init__(self, parent, identifier, grid_array):
        """Cell_Item is a super-class for Blob, Backgroun and Cell and should
        not be accessed directly.

        It takes these argument:

        @parent         The parent of the class

        @identifier     A id list (plate, row, column) so that it knows its
                        position.

        @grid_array     The first image_section (will initialize a filter
                        array of the same size.

        It has some functions:

        set_data_soruce Sets the image data array

        set_type        Checks and defines the type of cell item a thing is

        do_analysis     Runs analysis on a cell type, given that it has
                        previously been detected

        get_round_kernel    A function to get a binary array with a circle
                            in the center."""

        self._parent = parent

        if parent is not None:
            self.logger = self._parent.logger
        else:
            self.logger = logger.Log_Garbage_Collector()

        self.grid_array = grid_array.copy()
        self.filter_array = np.zeros(grid_array.shape, dtype=grid_array.dtype)

        self._identifier = identifier

        self.features = {}
        self._features_key_list = ['area', 'mean', 'median', 'IQR',
                    'IQR_mean', 'pixelsum']

        self.CELLITEM_TYPE = 0
        self.old_filter = None
        self.set_type()

    #
    # SET functions
    #

    def set_data_source(self, data_source):

        self.grid_array = data_source

        if self.grid_array.shape != self.filter_array.shape:

            self.logger.warning("GRID CELL " + \
                    "{0}: I just changed shape! Why?".format(
                    self._identifier))

            self.filter_array = np.zeros(self.grid_array.shape,
                                    dtype=self.grid_array.dtype)

    def set_type(self):

        """Empties the features-dictionary (as a precausion)
        and sets the cell item type.

        The function takes no argument"""

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


        CELLITEM_TYPEs:

        Blob            1
        Background      2
        Cell            3
        """

        if self.CELLITEM_TYPE == 0 or self.filter_array == None:

            self.features = dict()

            self.logger.warning("GRID CELL " + \
                    "{0}: Not properly initialized cell compartment".format(
                    self._identifier))

            return None

        self.features = {k: None for k in self._features_key_list}

        self.features['area'] = self.filter_array.sum()

        self.features['pixelsum'] = \
                self.grid_array[np.where(self.filter_array)].sum()

        if self.features['area'] == self.features['pixelsum'] or \
                                        self.features['area'] == 0:

            if self.features['area'] != 0:

                self.logger.warning("GRID CELL " + \
                        "{0}, seems to have all pixels value 1".format(
                        self._identifier))

            else:

                self.logger.warning("GRID CELL {0}, area is 0".format(
                        self._identifier))

            return None

        if self.features['area'] != 0:

            self.features['mean'] = self.features['pixelsum'] / \
                                            self.features['area']

            feature_array = self.grid_array[np.where(self.filter_array)]
            self.features['median'] = np.median(feature_array)
            self.features['IQR'] = mquantiles(feature_array, prob=[0.25, 0.75])

            try:

                self.features['IQR_mean'] = tmean(feature_array,
                                            self.features['IQR'])

            except:

                self.features['IQR_mean'] = None
                self.features['IQR'] = None

                debug.warning("GRID CELL %s, Failed to calculate IQR_mean," +\
                    " probably because IQR '%s' is empty." % \
                    ("unknown", str(self.features['IQR'])))

        else:

            self.features['mean'] = None
            self.features['median'] = None
            self.features['IQR'] = None
            self.features['IQR_mean'] = None

        if self.CELLITEM_TYPE == 1:

            try:

                self.features['centroid'] = center_of_mass(self.filter_array)

            except:

                self.features['centroid'] = None

            self.features['perimeter'] = None

#
# CLASS Blob
#


class Blob(Cell_Item):

    DEFAULT = 0
    ITERATIVE = 1
    THRESHOLD = 2

    def __init__(self, parent, identifier, grid_array, run_detect=True,
                    threshold=None, blob_detect='default',
                    image_color_logic="norm", center=None, radius=None):

        Cell_Item.__init__(self, parent, identifier, grid_array)

        self.threshold = threshold

        detect_types = {'default': self.DEFAULT,
            'iterative': self.ITERATIVE,
            'threshold': self.THRESHOLD}

        try:

            self.blob_detect = detect_types[blob_detect.lower()]

        except:

            self.blob_detect = self.DEFAULT

        self.old_trash = None
        self.trash_array = None
        self.image_color_logic = image_color_logic
        self._features_key_list += ['centroid', 'perimeter']

        self.histogram = hist.Histogram(self.grid_array, run_at_init=False)

        self.blob_recipe = rblob.Analysis_Recipe_Empty(self)
        rblob.Analysis_Recipe_Median_Filter(self, self.blob_recipe)
        rblob.Analysis_Threshold_Otsu(self, self.blob_recipe)
        rblob.Analysis_Recipe_Erode(self, self.blob_recipe)
        rblob.Analysis_Recipe_Dilate(self, self.blob_recipe)
        #rblob.Analysis_Recipe_Erode_Small(self, self.blob_recipe)
        #rblob.Analysis_Recipe_Erode_Conditional(self, self.blob_recipe)

        if run_detect:

            if center is not None and radius is not None:

                self.manual_detect(center, radius)

            elif self.blob_detect == self.THRESHOLD:

                self.threshold_detect()

            else:

                self.default_detect()

        self._debug_ticker = 0

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

        self.filter_array *= 0

        if rect:

            self.filter_array[rect[0][0]: rect[1][0],
                                rect[0][1]: rect[1][1]] = 1

        elif circle:

            raster = np.fromfunction(lambda i, j: 100 + 10 * i + j,
                            self.grid_array.shape, dtype=int)

            pts_iterator = points_in_circle(circle)  # , raster)

            for pt in pts_iterator:
                self.filter_array[pt] = 1

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

        if threshold is not None:

            if relative:

                self.threshold += threshold

            else:

                self.threshold = threshold
        else:

            if im is None:
                im = self.grid_array

            self.histogram.re_hist(im)
            self.threshold = hist.otsu(histogram=self.histogram)
    #
    # GET functions
    #

    def get_onion_values(self, A, A_filter, layer_size):
        """
        get_onion_value peals off bits of the A_filter and sums up
        what is left in A until nothing rematins in A_filter. At each
        layer it subtracts itself from the previous to become an onion.
        It returns a 2D array of sum and pixel count pairs.
        It leaves all sent parameters untouched...
        """

        onion_filter = A_filter.copy()
        onion = []

        while onion_filter.sum() > 0:

            onion.insert(0, [np.sum(A * onion_filter), onion_filter.sum()])

            if onion[0][0] <= 0:

                onion[0][0] = 1

            if len(onion) > 1:

                onion[1] = (np.log2(onion[1][0]) - np.log2(onion[0][0]),
                            onion[1][1] - onion[0][1])

            onion_filter = binary_erosion(onion_filter, iterations=layer_size)

        return np.asarray(onion)

    def get_diff(self, other_img, other_blob):
        """
        get_diff withdraws the other_img values from current image
        (a copy of it) superimposoing them using each blob-detection
        as reference point
        """

        cur_center = center_of_mass(self.filter_array)
        other_center = center_of_mass(other_blob)

        offset = np.round(np.asarray(other_center) - np.asarray(cur_center))

        if np.isnan(offset).sum() > 0:
            offset = np.zeros(2)

        return get_array_subtraction(other_img, self.grid_array, offset)

    def get_ideal_circle(self, c_array=None):
        """
        get_ideal_circle is a function that extracts the ideal
        circle from an array assuming that there's only one
        continious solid object in the array.

        It has one optional parameter:

        @c_array    An array to be analysed, if not passed
                    the current filter-array will be used instead.

        The function returns the following tuple:

            ( center_of_mass_position, radius )
        """

        if c_array is None:

            c_array = self.filter_array

        center_of_mass_position = center_of_mass(c_array)

        radius = (np.sum(c_array) / np.pi) ** 0.5

        return (center_of_mass_position, radius)

    def get_circularity(self, c_array=None):
        """
        get_circularity uses get_ideal_circle to make an abstract model
        of the object in c_array and passes this information to
        get_round_kernel producing the ideal circle as an array. This
        is subracted from the mass-center of the object in c_array.
        The differating pixels are summed and used as a measure of the
        circularity dividing it by the square root sum of pixels in the
        blob (to make the fraction independent for radius for near circular
        objects).

        The function takes one optional argument:

        @c_array        Array containing a blob, if nothing is passed then
                        self.filter_array will be used.

        The function returns a fraction value that estimates the
        circularity of the blob
        """

        if c_array is None:

            c_array = self.filter_array

        if c_array.sum() < 1:

            return 1000

        center_of_mass_position, radius = self.get_ideal_circle(c_array)

        radius = round(radius)

        perfect_blob = get_round_kernel(radius=radius)

        offset = [np.round(i[0] - i[1] / 2.0) for i in \
                zip(center_of_mass_position,  perfect_blob.shape)]

        diff_array = np.abs(get_array_subtraction(c_array,
                                    perfect_blob, offset))

        ###DEBUG CIRCULARITY
        #if self.grid_array.max() < 1000:
            #from matplotlib import pyplot as plt
            #plt.imshow(diff_array)
            #plt.show()
        ###DEBUG END

        return diff_array.sum() / (np.sqrt(c_array.sum()) * np.pi)

    #
    # DETECT functions
    #

    def detect(self, blob_detect=None, max_change_threshold=8,
                                remember_filter=False):
        """
        Generic wrapper function for blob-detection that calls the
        proper detection function and evaluates the results in comparison
        to the detected blob at time t+1

        Optional argument:

        @use_fallback_detection     If set, overrides the instance default

        @max_change_threshold       The max sum of differentiating pixels
                                    devided by old filters sum of pixels.
        """

        #DETECT BLOB ACCORDING TO METHOD
        if self.filter_array is not None:

            self.trash_array = np.zeros(self.filter_array.shape, dtype=bool)

        if blob_detect is None:

            blob_detect = self.blob_detect

        if blob_detect == self.DEFAULT:

            self.default_detect()

        elif blob_detect == self.ITERATIVE:

            self.iterative_threshold_detect()

        elif blob_detect == self.THRESHOLD:

            self.threshold_detect()

        else:

            self.default_detect()

        #DEAL WITH WHAT COULD BE DUST ETC -> TRASHED PIXELS
        if self.trash_array is not None:

            self.trash_array = np.zeros(self.filter_array.shape, dtype=bool)

        #COMPAIR WITH OLD FILTER
        if self.old_filter is not None:

            #DEMAND NOT TO LOOSE FILTER COMPLETELY
            if self.filter_array.sum() == 0:

                self.filter_array = self.old_filter.copy()

            #SHOULD BE MOVED/USE GENERAL FUNCTION AT MODlvl
            blob_diff = (np.abs(self.old_filter - self.filter_array)).sum()
            sqrt_of_oldsum = self.old_filter.sum() ** 0.5

            if blob_diff / float(sqrt_of_oldsum) > max_change_threshold:

                bad_diff = False

                if self.filter_array.sum() == 0 or self.old_filter.sum() == 0:

                    bad_diff = True

                else:

                    old_com = center_of_mass(self.old_filter)
                    new_com = center_of_mass(self.filter_array)

                    dim_1_offset = int(old_com[0] - new_com[0])
                    dim_2_offset = int(old_com[1] - new_com[1])

                    diff_filter = self.old_filter.copy()

                    if dim_1_offset > 0 and dim_2_offset > 0:

                        diff_filter = \
                            self.old_filter[dim_1_offset:, dim_2_offset:] -\
                            self.filter_array[:-dim_1_offset, :-dim_2_offset]

                    elif dim_1_offset < 0 and dim_2_offset < 0:

                        diff_filter = \
                            self.old_filter[: dim_1_offset, : dim_2_offset] -\
                            self.filter_array[-dim_1_offset:, -dim_2_offset:]

                    elif dim_1_offset > 0 and dim_2_offset < 0:

                        diff_filter = \
                            self.old_filter[dim_1_offset:, : dim_2_offset] -\
                            self.filter_array[:-dim_1_offset, -dim_2_offset:]

                    elif dim_1_offset < 0 and dim_2_offset > 0:

                        diff_filter = \
                            self.old_filter[: dim_1_offset, dim_2_offset:] -\
                            self.filter_array[-dim_1_offset:, :-dim_2_offset]

                    elif dim_1_offset == 0 and dim_2_offset < 0:

                        diff_filter = \
                            self.old_filter[:, : dim_2_offset] -\
                            self.filter_array[:, -dim_2_offset:]

                    elif dim_1_offset == 0 and dim_2_offset > 0:

                        diff_filter = \
                            self.old_filter[:, dim_2_offset:] -\
                            self.filter_array[:, :-dim_2_offset]

                    elif dim_1_offset < 0 and dim_2_offset == 0:

                        diff_filter = \
                            self.old_filter[: dim_1_offset, :] -\
                            self.filter_array[-dim_1_offset:, :]

                    elif dim_1_offset > 0 and dim_2_offset == 0:
                        diff_filter = \
                            self.old_filter[dim_1_offset:, :] -\
                            self.filter_array[:-dim_1_offset, :]

                    else:
                        diff_filter = self.old_filter - self.filter_array

                    blob_diff = (np.abs(diff_filter)).sum()

                    if blob_diff / float(sqrt_of_oldsum) > \
                                        max_change_threshold:

                        bad_diff = True

                if bad_diff:

                    self.filter_array = self.old_filter.copy()

                    if self.old_trash is not None:

                        self.trash_array = self.old_trash.copy()

                    self.logger.warning("GRID CELL " + \
                            ("{0}, Blob detection gone bad, using old " + \
                            "(Error: {1:.2f})").format(self._identifier,
                            blob_diff / float(sqrt_of_oldsum)))

        #IF FILTER SHOULD BE REMEMBERED THEN DO SO
        if remember_filter:

            self.old_filter = self.filter_array.copy()

            if self.trash_array is not None:

                self.old_trash = self.trash_array.copy()

    def iterative_threshold_detect(self):

        #De-noising the image with a smooth
        grid_array = gaussian_filter(self.grid_array, 2)

        threshold = 1
        self.threshold_detect(im=grid_array, threshold=threshold)

        while self.get_circularity() > 10 and threshold < 124:

            threshold *= 1.5
            self.threshold_detect(im=grid_array, threshold=threshold)

    def threshold_detect(self, im=None, threshold=None, color_logic=None):
        """
        If there is a threshold previously set, this will be used to
        detect blob by accepting everythin above threshold as the blob.

        If no threshold has been set, threshold is calculated using
        Otsu on the histogram of the image-section.

        Function takes one optional argument:

        @im             Optional alternative image source
        """

        if self.threshold == None or threshold is not None:

            self.set_threshold(im=im, threshold=threshold)

        if im is None:

            im = self.grid_array

        if color_logic is None:

            color_logic = self.image_color_logic

        self.filter_array *= 0

        if color_logic == "inv":

            self.filter_array[np.where(im < self.threshold)] = 1

        else:

            self.filter_array[np.where(im > self.threshold)] = 1

    def manual_detect(self, center, radius):

        self.filter_array *= 0

        stencil = get_round_kernel(int(np.round(radius)))
        x_size = (stencil.shape[0] - 1) / 2
        y_size = (stencil.shape[1] - 1) / 2

        if stencil.shape == \
                self.filter_array[center[0] - x_size: center[0] + x_size + 1,
                center[1] - y_size: center[1] + y_size + 1].shape:

            self.filter_array[center[0] - x_size: center[0] + x_size + 1,
                center[1] - y_size: center[1] + y_size + 1] += stencil

        else:

            self.default_detect()

    def default_detect(self):

        self.blob_recipe.set_reference_image(self.grid_array)
        self.blob_recipe.analyse()
        self.keep_best_blob()

    def get_candidate_blob_ranks(self):

        label_array, number_of_labels = label(self.filter_array)
        qualities = []
        c_o_m = {}

        if number_of_labels > 0:

            for item in xrange(number_of_labels):

                cur_item = (label_array == (item + 1))

                c_o_m[item] = center_of_mass(cur_item)

                cur_pxs = np.sum(cur_item)

                oneD = np.where(np.sum(cur_item, 1) > 0)[0]
                dim1 = oneD[-1] - oneD[0]
                oneD = np.where(np.sum(cur_item, 0) > 0)[0]
                dim2 = oneD[-1] - oneD[0]

                if dim1 > dim2:

                    qualities.append(cur_pxs * dim2 / float(dim1))

                else:

                    qualities.append(cur_pxs * dim1 / float(dim2))

        return number_of_labels, qualities, c_o_m, label_array

    def keep_best_blob(self):
        """Evaluates all blobs detected and keeps the best one"""

        number_of_labels, qualities, c_o_m, label_array = \
                                self.get_candidate_blob_ranks()

        if number_of_labels > 0:

            q_best = np.asarray(qualities).argmax() + 1

            self.filter_array = (label_array == q_best)

            composite_blob = [q_best]
            composite_trash = []

            for item in xrange(number_of_labels):

                if self.filter_array[tuple(map(round, c_o_m[item]))]:

                    composite_blob.append(item + 1)

                else:

                    composite_trash.append(item + 1)

            #if len(composite_blob) > 1
            self.filter_array = np.in1d(label_array,
                    np.array(composite_blob)).reshape(self.filter_array.shape,
                    order='C')

            #self.trash_array = (label_array > 0) * (label_array != q_best +1)
            self.trash_array = np.in1d(label_array,
                    np.array(composite_trash)).reshape(self.filter_array.shape,
                    order='C')

#
# CLASSES Background (inverse blob area)
#


class Background(Cell_Item):

    def __init__(self, parent, identifier, grid_array, blob, run_detect=True):

        Cell_Item.__init__(self, parent, identifier, grid_array)

        if isinstance(blob, Blob):

            self.blob = blob

        else:

            self.blob = None

        if run_detect:

            self.detect()

    def detect(self, **kwargs):
        """
        detect finds the background

        It is assumed that the background is the inverse
        of the blob. Therefore this function only runs after
        the detect function has been run on blob.

        Function takes no arguments (**kwargs just there to keep interface)
        """

        if self.blob and self.blob.filter_array != None:

            self.filter_array[:, :] = 1

            self.filter_array[np.where(self.blob.filter_array)] = 0

            self.filter_array[np.where(self.blob.trash_array)] = 0

            self.filter_array = binary_erosion(self.filter_array,
                                iterations=6, border_value=1)

        else:

            self.logger.warning(("GRID CELL {0}, blob was not set, " + \
                    "thus background is wrong").format(self._identifier))

#
# CLASSES Cell (entire area)
#


class Cell(Cell_Item):

    def __init__(self, parent, identifier, grid_array,
                        run_detect=True, threshold=-1):

        Cell_Item.__init__(self, parent, identifier, grid_array)

        self.threshold = threshold

        self.filter_array[:, :] = 1

        if run_detect:

            self.detect()

    def detect(self, **kwargs):
        """
        detect makes a filter that is true for the full area

        The function takes no argument.
        """
        pass
