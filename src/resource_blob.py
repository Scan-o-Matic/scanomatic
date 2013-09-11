#!/usr/bin/env python
"""
Resource for blob detection recepies
"""
__author__ = "Martin Zackrisson, jetxee"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation,\
    gaussian_filter, median_filter
from skimage import filter as ski_filter

#
# SCANNOMATIC LIBRARIES
#

#import resource_histogram as resource_histogram

#
# CLASSES Cell_Item
#


class Analysis_Recipe_Abstraction(object):
    """Holds an instruction and/or a list of subinstructions."""

    def __init__(self, parent=None, description=""):

        self.analysis_order = [self]
        self.description = description

        if parent is not None:

            parent.add_anlysis(self)

    def __str__(self):

        return self.description

    def __repr__(self):

        return "<{0} {1}>".format(id(self), self.description)

    """
    def set_reference_image(self, im, inplace=False, enforce_self=False,
                                                        do_copy=True):

        if enforce_self or self.parent is None:

            dest = self

        else:

            dest = self.parent

        if self._analysis_image is None and inplace:

            inplace = False

        if inplace:

            dest._analysis_image[:, :] = im

        else:

            if do_copy:

                im = im.copy()

            dest._analysis_image = im

        if enforce_self == False and self.analysis_order != [self]:

            self.set_reference_image(im, inplace=inplace, enforce_self=True,
                                        do_copy=False)
    """

    def analyse(self, im, filter_array, baseLvl=True):

        if baseLvl:
            filter_array[...] = im.copy()

        for a in self.analysis_order:

            if a is self:

                self._do(filter_array)

            else:

                a.analyse(im, filter_array, baseLvl=False)

    def add_anlysis(self, a, pos=-1):

        if pos == -1:

            self.analysis_order.append(a)

        else:

            self.analysis_order.insert(pos, a)

    def _do(self, filter_array):

        pass


class Analysis_Recipe_Empty(Analysis_Recipe_Abstraction):

    def __init__(self, parent=None):

        super(Analysis_Recipe_Empty, self).__init__(
            parent, description="Recipe")

        self.analysis_order = []


class Analysis_Threshold_Otsu(Analysis_Recipe_Abstraction):

    def __init__(self, parent):

        super(Analysis_Threshold_Otsu, self).__init__(
            parent, description="Otsu Threshold")

    def _do(self, filter_array):

        """
        threshold = resource_histogram.otsu(
            histogram=resource_histogram.Histogram(
                filter_array, run_at_init=True))
        """

        filter_array[...] = filter_array > ski_filter.threshold_otsu(
            filter_array)


class Analysis_Recipe_Erode(Analysis_Recipe_Abstraction):

    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]])

    def __init__(self, parent):

        super(Analysis_Recipe_Erode, self).__init__(
            parent, description="Binary Erode")

    def _do(self, filter_array):

        #Erosion kernel
        #kernel = get_round_kernel(radius=2)
        #print kernel.astype(int)
        #print "***Erosion kernel ready"

        binary_erosion(filter_array, iterations=3, output=filter_array)


class Analysis_Recipe_Erode_Small(Analysis_Recipe_Abstraction):

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    def __init__(self, parent):

        super(Analysis_Recipe_Erode_Small, self).__init__(
            parent, description="Binary Erode (small)")

    def _do(self, filter_array):

        binary_erosion(filter_array, origin=(1, 1),
                       structure=self.kernel, output=filter_array)


"""
class Analysis_Recipe_Erode_Conditional(Analysis_Recipe_Abstraction):

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    threshold = 4

    def __init__(self, parent):

        super(Analysis_Recipe_Erode_Conditional, self).__init__(
            parent, description="Binary Erode")

    def _do(self, filter_array):

        fa = self.grid_cell.filter_array

        if np.median(im[np.where(fa == 1)]) / \
                float(np.median(im[np.where(fa == 0)])) < self.threshold:

            self.grid_cell.filter_array = binary_erosion(
                                fa,
                                origin=(1,1),
                                structure=self.kernel)
"""


class Analysis_Recipe_Dilate(Analysis_Recipe_Abstraction):

    kernel = np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]])

    def __init__(self, parent):

        super(Analysis_Recipe_Dilate, self).__init__(
            parent, description="Binary Dilate")

    def _do(self, filter_array):

        #Erosion kernel
        #kernel = get_round_kernel(radius=2)
        #print kernel.astype(int)
        #print "***Erosion kernel ready"

        binary_dilation(filter_array, iterations=4, output=filter_array)
        #origin=(3,3),
        #structure=self.kernel)


class Analysis_Recipe_Gauss_2(Analysis_Recipe_Abstraction):

    def __init__(self, parent):

        super(Analysis_Recipe_Gauss_2, self).__init__(
            parent, description="Gaussian size 2")

    def _do(self, filter_array):

        gaussian_filter(filter_array, 2, output=filter_array)


class Analysis_Recipe_Median_Filter(Analysis_Recipe_Abstraction):

    def __init__(self, parent):

        super(Analysis_Recipe_Median_Filter, self).__init__(
            parent, description="Median Filter")

    def _do(self, filter_array):

        median_filter(filter_array, size=(3, 3), mode="nearest",
                      output=filter_array)
