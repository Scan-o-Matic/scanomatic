#
# DEPENDENCIES
#

import numpy as np
import operator
from enum import Enum
from scipy.ndimage import (
    binary_erosion, center_of_mass, label, gaussian_filter)
#
# SCANNOMATIC LIBRARIES
#

import scanomatic.image_analysis.histogram as histogram
import scanomatic.image_analysis.blob as blob
from scanomatic.models.factories.analysis_factories import (
    AnalysisFeaturesFactory)
from scanomatic.models.analysis_model import MEASURES
from scanomatic.generics.maths import mid50_mean as iqr_mean, quantiles_stable

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


def get_round_kernel(radius=6.0, outline=False):

    round_kernel = np.zeros(((radius + 1) * 2 + 1, (radius + 1) * 2 + 1),
                            dtype=np.bool)

    center_offset = radius + 1

    y, x = np.ogrid[-radius: radius, -radius: radius]

    if outline:

        index = radius ** 2 - 1 <= x ** 2 + y ** 2 <= radius ** 2 + 2

    else:

        index = x ** 2 + y ** 2 <= radius ** 2

    round_kernel[center_offset - radius: center_offset + radius,
                 center_offset - radius: center_offset + radius][index] = True

    return round_kernel


def get_array_subtraction(array_one, array_two, offset, output=None):
    """Makes offsetted subtractions for A1 - A2 independent of sizes

    If output is supplied it will be fed directly into it, else,
    it will just return a new array.
    """

    o1_low = offset[0]
    o2_low = offset[1]

    o1_high = o1_low + array_two.shape[0]
    o2_high = o2_low + array_two.shape[1]

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

    if o1_high > array_one.shape[0]:

        b1_high = array_two.shape[0] - (o1_high - array_one.shape[0])
        o1_high = array_one.shape[0]

    else:

        b1_high = array_two.shape[0]

    if o2_high > array_one.shape[1]:

        b2_high = array_two.shape[1] - (o2_high - array_one.shape[1])
        o2_high = array_one.shape[1]

    else:

        b2_high = array_two.shape[1]

    if output is None:

        diff_array = array_one.copy()

        diff_array[o1_low: o1_high, o2_low: o2_high] -= \
            array_two[b1_low: b1_high, b2_low: b2_high]

        return diff_array

    else:

        output[o1_low: o1_high, o2_low: o2_high] = \
            array_one[o1_low: o1_high, o2_low: o2_high] - \
            array_two[b1_low: b1_high, b2_low: b2_high]

#
# CLASSES Cell_Item
#


class CellItem(object):

    def __init__(self, identifier, grid_array):
        """Cell_Item is a super-class for Blob, Background and Cell and should
        not be accessed directly.

        It takes these argument:

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

        self.grid_array = grid_array.copy()
        self.filter_array = np.zeros(grid_array.shape, dtype=np.bool)

        self._identifier = identifier
        self._compartment_type = identifier[-1]
        self.features = AnalysisFeaturesFactory.create(index=self._compartment_type, data={})

        self._features_key_list = [
            MEASURES.Count,
            MEASURES.Mean,
            MEASURES.Median,
            MEASURES.IQR,
            MEASURES.IQR_Mean,
            MEASURES.Sum]

        self.features.shape = (len(self._features_key_list),)

        self.old_filter = None

    #
    # SET functions
    #

    def set_data_source(self, data_source):

        self.grid_array = data_source

        if self.grid_array.shape != self.filter_array.shape:

            self.filter_array = np.zeros(self.grid_array.shape,
                                         dtype=np.bool)

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

        feature_data = self.features.data
        """:type : dict[scanomatic.models.analysis_model.MEASUERS|object]"""

        if self.filter_array is None or len(self._features_key_list) == 0:

            return

        feature_array = None
        feature_data[MEASURES.Count] = self.filter_array.sum()

        feature_data[MEASURES.Sum] = self.grid_array[np.where(self.filter_array)].sum()

        if feature_data[MEASURES.Count] == feature_data[MEASURES.Sum] or feature_data[MEASURES.Count] == 0:

            if feature_data[MEASURES.Count] == 0:

                print "GCdissect", self._identifier, "No blob"

            else:

                print "GCdissect", self._identifier, "No background"

            feature_data.clear()

        else:

            feature_data[MEASURES.Mean] = feature_data[MEASURES.Sum] / \
                feature_data[MEASURES.Count]

            if (MEASURES.Median in self._features_key_list or
                        MEASURES.IQR in self._features_key_list or
                        MEASURES.IQR_Mean in self._features_key_list):

                feature_array = self.grid_array[np.where(self.filter_array)]

            if MEASURES.Median in self._features_key_list:
                feature_data[MEASURES.Median] = np.median(feature_array)

            if MEASURES.IQR in self._features_key_list or MEASURES.IQR_Mean in self._features_key_list:
                feature_data[MEASURES.IQR] = quantiles_stable(feature_array)
                # mquantiles(feature_array, prob=[0.25, 0.75])

                try:

                    feature_data[MEASURES.IQR_Mean] = iqr_mean(feature_array)
                    # tmean(feature_array, feature_data['IQR'])

                except:

                    feature_data[MEASURES.IQR_Mean] = None
                    feature_data[MEASURES.IQR] = None

            if MEASURES.Centroid in self._features_key_list:

                try:

                    feature_data[MEASURES.Centroid] = center_of_mass(self.filter_array)

                except:

                    feature_data[MEASURES.Centroid] = None

            if MEASURES.Perimeter in self._features_key_list:
                feature_data[MEASURES.Perimeter] = None

#
# CLASS Blob
#


def get_onion_values(array, array_filter, layer_size):
    """
    get_onion_value peals off bits of the A_filter and sums up
    what is left in A until nothing rematins in A_filter. At each
    layer it subtracts itself from the previous to become an onion.
    It returns a 2D array of sum and pixel count pairs.
    It leaves all sent parameters untouched...
    """

    onion_filter = array_filter.copy()
    onion = []

    while onion_filter.sum() > 0:

        onion.insert(0, [np.sum(array * onion_filter), onion_filter.sum()])

        if onion[0][0] <= 0:

            onion[0][0] = 1

        if len(onion) > 1:

            onion[1] = (np.log2(onion[1][0]) - np.log2(onion[0][0]),
                        onion[1][1] - onion[0][1])

        onion_filter = binary_erosion(onion_filter, iterations=layer_size)

    return np.asarray(onion)


class BlobDetectionTypes(Enum):

    DEFAULT = 0
    ITERATIVE = 1
    THRESHOLD = 2


class Blob(CellItem):

    BLOB_RECIPE = blob.AnalysisRecipeEmpty()
    blob.AnalysisRecipeMedianFilter(BLOB_RECIPE)
    blob.AnalysisThresholdOtsu(BLOB_RECIPE, threshold_unit_adjust=0.5)
    blob.AnalysisRecipeDilate(BLOB_RECIPE, iterations=2)

    def __init__(self, identifier, grid_array, run_detect=True, threshold=None, blob_detect=BlobDetectionTypes.DEFAULT,
                 image_color_logic="norm", center=None, radius=None):

        CellItem.__init__(self, identifier, grid_array)

        self.threshold = threshold

        if not isinstance(blob_detect, BlobDetectionTypes):
            try:

                blob_detect = BlobDetectionTypes[blob_detect.upper()]

            except KeyError:

                blob_detect = BlobDetectionTypes.DEFAULT

        if blob_detect is BlobDetectionTypes.THRESHOLD:
            self.detect_function = self.threshold
        elif blob_detect is BlobDetectionTypes.ITERATIVE:
            self.detect_function = self.iterative_threshold_detect
        else:
            self.detect_function = self.default_detect

        self.old_trash = None
        self.trash_array = None
        self.image_color_logic = image_color_logic
        self._features_key_list += [MEASURES.Centroid, MEASURES.Perimeter]
        self.features.shape = (len(self._features_key_list),)

        self.histogram = histogram.Histogram(self.grid_array, run_at_init=False)

        if run_detect:

            if center is not None and radius is not None:

                self.manual_detect(center, radius)

            else:

                self.detect_function()

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

        self.filter_array[...] = False

        if rect:

            self.filter_array[rect[0][0]: rect[1][0],
                              rect[0][1]: rect[1][1]] = True

        elif circle:

            """
            raster = np.fromfunction(
                lambda i, j: 100 + 10 * i + j,
                self.grid_array.shape, dtype=int)
            """

            pts_iterator = points_in_circle(circle)  # , raster)

            for pt in pts_iterator:
                self.filter_array[pt] = True

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
            self.threshold = histogram.otsu(histogram=self.histogram)
    #
    # GET functions
    #

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

        return center_of_mass_position, radius

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

        offset = [np.round(i[0] - i[1] / 2.0) for i in
                  zip(center_of_mass_position,  perfect_blob.shape)]

        diff_array = np.abs(get_array_subtraction(
            c_array, perfect_blob, offset))

        return diff_array.sum() / (np.sqrt(c_array.sum()) * np.pi)

    #
    # DETECT functions
    #

    def detect(self, detect_type=None, max_change_threshold=8,
               remember_filter=True, remember_trash=False):
        """
        Generic wrapper function for blob-detection that calls the
        proper detection function and evaluates the results in comparison
        to the detected blob at time t+1

        Optional argument:

        @use_fallback_detection     If set, overrides the instance default

        @max_change_threshold       The max sum of differentiating pixels
                                    devided by old filters sum of pixels.
        """

        if self.filter_array is not None:

            self.trash_array = np.zeros(self.filter_array.shape, dtype=np.bool)

        if detect_type is None:

            self.detect_function()
        else:

            if detect_type is BlobDetectionTypes.ITERATIVE:

                self.iterative_threshold_detect()

            elif detect_type is BlobDetectionTypes.THRESHOLD:

                self.threshold_detect()

            else:

                self.default_detect()

        if self.trash_array is None:

            self.trash_array = np.zeros(self.filter_array.shape, dtype=np.bool)

        if self.old_filter is not None:

            if self.filter_array.sum() == 0:
                self.filter_array = self.old_filter.copy()

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

                    if dim_1_offset > 0 and dim_2_offset > 0:

                        diff_filter = \
                            self.old_filter[dim_1_offset:, dim_2_offset:] -\
                            self.filter_array[:-dim_1_offset, :-dim_2_offset]

                    elif dim_1_offset < 0 and dim_2_offset < 0:

                        diff_filter = \
                            self.old_filter[: dim_1_offset, : dim_2_offset] - \
                            self.filter_array[-dim_1_offset:, -dim_2_offset:]

                    elif dim_1_offset > 0 > dim_2_offset:

                        diff_filter = \
                            self.old_filter[dim_1_offset:, : dim_2_offset] - \
                            self.filter_array[:-dim_1_offset, -dim_2_offset:]

                    elif dim_1_offset < 0 < dim_2_offset:

                        diff_filter = \
                            self.old_filter[: dim_1_offset, dim_2_offset:] - \
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

                    elif dim_1_offset > 0 == dim_2_offset:
                        diff_filter = \
                            self.old_filter[dim_1_offset:, :] -\
                            self.filter_array[:-dim_1_offset, :]

                    else:
                        diff_filter = self.old_filter - self.filter_array

                    blob_diff = diff_filter.sum()

                    if blob_diff / float(sqrt_of_oldsum) > max_change_threshold:

                        bad_diff = True

                if bad_diff:

                    self.filter_array = self.old_filter.copy()

                    if self.old_trash is not None:

                        self.trash_array = self.old_trash.copy()

        if remember_filter:

            self.old_filter = self.filter_array.copy()

        if remember_trash:

            if self.trash_array is not None:

                self.old_trash = self.trash_array.copy()

    def iterative_threshold_detect(self):

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

        if self.threshold is None or threshold is not None:

            self.set_threshold(im=im, threshold=threshold)

        if im is None:

            im = self.grid_array

        if color_logic is None:

            color_logic = self.image_color_logic

        self.filter_array[...] = False

        if color_logic == "inv":

            self.filter_array[np.where(im < self.threshold)] = True

        else:

            self.filter_array[np.where(im > self.threshold)] = True

    def manual_detect(self, center, radius):

        self.filter_array[...] = False

        stencil = get_round_kernel(int(np.round(radius)))
        x_size = (stencil.shape[0] - 1) / 2
        y_size = (stencil.shape[1] - 1) / 2
        center = map(int, map(round, center))

        if (self.filter_array.shape[0] > center[0] + x_size + 1
                and center[0] - x_size >= 0):

            x_slice = slice(center[0] - x_size, center[0] + x_size + 1, None)
            x_stencil_slice = slice(None, None, None)

        elif center[0] - x_size < 0:

            x_slice = slice(None, center[0] + x_size + 1, None)
            x_stencil_slice = slice(stencil.shape[0] - (center[0] + x_size + 1), None, None)

        else:

            x_slice = slice(center[0] - x_size, None, None)
            x_stencil_slice = slice(None, self.filter_array.shape[0] - center[0] + x_size, None)

        if (self.filter_array.shape[1] > center[1] + y_size + 1
                and center[1] - y_size >= 0):

            y_slice = slice(center[1] - y_size, center[1] + y_size + 1, None)
            y_stencil_slice = slice(None, None, None)

        elif center[1] - y_size < 0:

            y_slice = slice(None, center[1] + y_size + 1, None)
            y_stencil_slice = slice(stencil.shape[1] - (center[1] + y_size + 1), None, None)

        else:

            y_slice = slice(center[1] - y_size, None, None)
            y_stencil_slice = slice(None, self.filter_array.shape[1] - center[1] + y_size, None)

        self.filter_array[(x_slice, y_slice)] += stencil[(x_stencil_slice, y_stencil_slice)]

    def default_detect(self):

        if self.grid_array.size:
            self.BLOB_RECIPE.analyse(self.grid_array, self.filter_array)
            self.keep_best_blob()

    def get_candidate_blob_ranks(self):

        label_array, number_of_labels = label(self.filter_array)
        qualities = {}
        centre_of_masses = {}

        if number_of_labels > 0:

            for label_value in xrange(1, number_of_labels + 1):

                current_label_filter = label_array == label_value

                if current_label_filter.sum() == 0:
                    continue

                centre_of_masses[label_value] = center_of_mass(current_label_filter)

                area = np.sum(current_label_filter)

                dim_extents = [-1, -1]
                for dim in range(2):
                    over_axis_sum = np.where(np.sum(current_label_filter, axis=dim) > 0)
                    dim_extents[dim] = max(*over_axis_sum) - min(*over_axis_sum) + 1.0

                qualities[label_value] = (area * min(dim_extents) / max(dim_extents))

        return number_of_labels, qualities, centre_of_masses, label_array

    # noinspection PyTypeChecker
    def keep_best_blob(self):
        """Evaluates all blobs detected and keeps the best one"""

        _, qualities, centre_of_masses, label_array = self.get_candidate_blob_ranks()

        if qualities:

            quality_order = zip(*sorted(qualities.iteritems(), key=operator.itemgetter(1)))[0][::-1]
            best_quality_label = quality_order[0]

            self.filter_array = label_array == best_quality_label

            composite_blob = [best_quality_label]
            composite_trash = []

            for item_label in quality_order[1:]:

                if self.filter_array[tuple(map(int, map(round, centre_of_masses[item_label])))]:

                    composite_blob.append(item_label)

                else:

                    composite_trash.append(item_label)

            self.filter_array = np.in1d(
                label_array, np.array(composite_blob)).reshape(self.filter_array.shape)

            self.trash_array = np.in1d(
                label_array, np.array(composite_trash)).reshape(self.filter_array.shape)

#
# CLASSES Background (inverse blob area)
#


class Background(CellItem):

    def __init__(self, identifier, grid_array, blob_instance, run_detect=True):

        CellItem.__init__(self, identifier, grid_array)

        if isinstance(blob_instance, Blob):

            self.blob = blob_instance

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

        if self.blob and self.blob.filter_array is not None:

            self.filter_array[...] = True

            self.filter_array[np.where(self.blob.filter_array)] = False

            self.filter_array[np.where(self.blob.trash_array)] = False

            self.filter_array = binary_erosion(
                self.filter_array, iterations=3, border_value=1)

        else:

            print "BG", self._identifier, "no blob"


#
# CLASSES Cell (entire area)
#


class Cell(CellItem):

    def __init__(self, identifier, grid_array,
                 run_detect=True, threshold=-1):

        CellItem.__init__(self, identifier, grid_array)

        self.threshold = threshold

        self.filter_array[...] = True

        if run_detect:

            self.detect()

    @staticmethod
    def detect(**kwargs):
        """
        detect makes a filter that is true for the full area

        The function takes no argument.
        """
        pass
