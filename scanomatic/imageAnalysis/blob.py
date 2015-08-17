#!/usr/bin/env python
"""
Resource for blob detection recepies
"""
__author__ = "Martin Zackrisson, jetxee"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
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

#
# CLASSES Cell_Item
#


class AnalysisRecipeAbstraction(object):
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

    def analyse(self, im, filter_array, base_level=True):

        if base_level:
            filter_array[...] = im.copy()

        for a in self.analysis_order:

            if a is self:

                self._do(filter_array)

            else:

                a.analyse(im, filter_array, base_level=False)

    def add_anlysis(self, a, pos=-1):

        if pos == -1:

            self.analysis_order.append(a)

        else:

            self.analysis_order.insert(pos, a)

    def _do(self, filter_array):

        pass


class AnalysisRecipeEmpty(AnalysisRecipeAbstraction):

    def __init__(self, parent=None):

        super(AnalysisRecipeEmpty, self).__init__(
            parent, description="Recipe")

        self.analysis_order = []


class AnalysisThresholdOtsu(AnalysisRecipeAbstraction):

    def __init__(self, parent, threshold_unit_adjust=0.0):

        super(AnalysisThresholdOtsu, self).__init__(parent, description="Otsu Threshold")

        self._thresholdUnitAdjust = threshold_unit_adjust

    def _do(self, filter_array):

        try:
            filter_array[...] = filter_array > ski_filter.threshold_otsu(
                filter_array) + self._thresholdUnitAdjust
        except ValueError:
            filter_array[...] = 0

class AnalysisRecipeErode(AnalysisRecipeAbstraction):

    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]])

    def __init__(self, parent):

        super(AnalysisRecipeErode, self).__init__(
            parent, description="Binary Erode")

    def _do(self, filter_array):

        filter_array[...] = binary_erosion(filter_array, iterations=3)


class AnalysisRecipeErodeSmall(AnalysisRecipeAbstraction):

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    def __init__(self, parent):

        super(AnalysisRecipeErodeSmall, self).__init__(
            parent, description="Binary Erode (small)")

    def _do(self, filter_array):

        filter_array[...] = binary_erosion(filter_array, origin=(1, 1), structure=self.kernel)


class AnalysisRecipeDilate(AnalysisRecipeAbstraction):

    kernel = np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]])

    def __init__(self, parent, iterations=4):

        super(AnalysisRecipeDilate, self).__init__(
            parent, description="Binary Dilate")

        self._iterations = iterations

    def _do(self, filter_array):

        filter_array[...] = binary_dilation(filter_array, iterations=self._iterations)


class AnalysisRecipeGauss2(AnalysisRecipeAbstraction):

    def __init__(self, parent):

        super(AnalysisRecipeGauss2, self).__init__(
            parent, description="Gaussian size 2")

    def _do(self, filter_array):

        gaussian_filter(filter_array, 2, output=filter_array)


class AnalysisRecipeMedianFilter(AnalysisRecipeAbstraction):

    def __init__(self, parent):

        super(AnalysisRecipeMedianFilter, self).__init__(
            parent, description="Median Filter")

    def _do(self, filter_array):

        median_filter(filter_array, size=(3, 3), mode="nearest",
                      output=filter_array)
