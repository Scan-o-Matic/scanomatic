#!/usr/bin/env python
"""Resource module for handling the aquired images."""

__author__ = "Martin Zackrisson, Andreas Skyman"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Andreas Skyman"]
__license__ = "GPL v3.0"
__version__ = "0.994"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#

from scipy.ndimage import zoom
from scipy.signal import fftconvolve
#from scipy.optimize import fsolve
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

#
# SCANNOMATIC DEPENDENCIES
#

import resource_signal as r_signal
import resource_grayscale

#
# GLOBALS
#

DEFAULT_GRAYSCALE = resource_grayscale.getDefualtGrayscale()

GRAYSCALE_NAMES = resource_grayscale.getGrayscales()

GRAYSCALES = {gsName: resource_grayscale.getGrayscale(gsName) for
              gsName in GRAYSCALE_NAMES}

_logger = logging.getLogger("Resource Image")
'''
DEFAULT_GRAYSCALE = 'Kodak'

GRAYSCALES = {
    'Kodak': {
        'targets': [0, 2, 4, 6, 10, 14, 18, 22, 26, 30, 34, 38,
                    42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82],
        'width': 55,
        'min_width': 30,
        'sections': 23,
        'lower_than_half_width': 350,
        'higher_than_half_width': 150,
        'length': 28.3,  # 28.57 was previous
    },
    'SilverFast': {
        'targets': [0, 2, 4, 6, 10, 14, 18, 22, 26, 30, 34, 38,
                    42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82],
        'width': 58,
        'min_width': 30,
        'sections': 23,
        'lower_than_half_width': 350,
        'higher_than_half_width': 150,
        'length': 29.565217391,
    }
}

"""
        'targets': [82, 78, 74, 70, 66, 62, 58, 54, 50, 46, 42,
                    38, 34, 30, 26, 22, 18, 14, 10, 6, 4, 2, 0],
"""

GRAYSCALE_SCALABLE = ('width', 'min_width', 'lower_than_half_width',
                      'higher_than_half_width', 'length')
'''

#
# FUNCTIONS
#


def Quick_Scale_To(source_path, target_path, source_dpi=600, target_dpi=150):

    small_im = Quick_Scale_To_im(source_path, source_dpi=source_dpi,
                                 target_dpi=target_dpi)

    try:

        np.save(target_path, small_im)

    except:

        _logger.error("Could not save scaled down image")

        return -1


def Quick_Scale_To_im(path=None, im=None, source_dpi=600, target_dpi=150,
                      scale=None):

    if im is None:

        try:

            im = plt.imread(path)

        except:

            _logger.error("Could not open source")

            return -1

    if scale is None:
        scale = target_dpi / float(source_dpi)

    small_im = zoom(im, scale, order=1)

    return small_im


class Image_Transpose(object):

    def __init__(self, sourceValues=None, targetValues=None, polyCoeffs=None):

        self._logger = logging.getLogger("Image Transpose")
        self._source = sourceValues
        self._target = targetValues
        self._polyCoeffs = polyCoeffs

        if (self._polyCoeffs is None and self._target is not None and
                self._source is not None):

            try:
                self._polyCoeffs = np.polyfit(self._source, self._target, 3)
            except Exception, e:
                self._logger.critical(
                    "Could not produce polynomial from source " +
                    "{0} and target {1}".format(self._source, self._target))

                raise e

        if self._polyCoeffs is not None:
            self._poly = np.poly1d(self._polyCoeffs)
        else:
            errorCause = ""
            if self._source is None:
                errorCause += "No source "
            if self._target is None:
                errorCause += "No target "
            if self._polyCoeffs is None:
                errorCause += "No Coefficients"
            raise Exception(
                "Polynomial not initiated; can't transpose image: {0}".format(
                    errorCause))

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def coefficients(self):
        return self._polyCoeffs

    @property
    def polynomial(self):
        return self._poly

    def __call__(self, im):

        return self._poly(im)


'''
class Image_Transpose(object):

    def __init__(self, *args, **kwargs):

        #So that p3 won't fail
        self.gs_a = 0
        self.gs_b = 0
        self.gs_c = 1
        self.gs_d = 0

        self.gs = None
        self.gs_targets = None

        self.set_matrix(*args, **kwargs)

    def _get_p3(self, x):
        """
            returns the solution to:

                self.gs_a * x^3 + self.gs_b * x^2 + self.gs_c * x + self.gs_d

        """

        p = self.gs_a * (x ** 3) + self.gs_b * (x ** 2) + \
            self.gs_c * x + self.gs_d

        return p

    def set_matrix(self, gs_values=None, gs_fit=None,
                   gs_indices=None, y_max=255, fix_axis=False):
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

        @y_max      An int for the upper bound of the solution
                    lower will always be 0

        @fix_axis   An optional possibility to fix the gs-axis,
                    else it will be made increasing (transformed with
                    -1 if not). Lowest value will also be set to 0,
                    assuming a continious series.

        The function returns a list of transformation values
        """

        if gs_values is not None:

            #Create value - indices if not supplied
            if gs_indices is None:

                gs_indices = range(len(gs_values))

            #Make it increasing
            if gs_indices[0] > gs_indices[-1]:

                gs_indices = map(lambda x: x * -1, gs_indices)

            #Move it to zero
            if gs_indices[0] != 0:

                gs_indices = map(lambda x: x - gs_indices[0], gs_indices)

            #Solve the polynomial
            p = np.poly1d(np.polyfit(gs_indices, gs_values, 3))

            self.gs_a = p.c[0]
            self.gs_b = p.c[1]
            self.gs_c = p.c[2]
            #self.gs_d gets reset each time below

            tf_matrix = np.zeros((y_max + 1))
            for i in xrange(256):

                #moving the line along y-axis
                self.gs_d = p.c[3] - i
                x = fsolve(self._get_p3, gs_values[0])

                #setting it back to get the values
                tf_matrix[i] = x

        else:

            tf_matrix = []

            for y in range(y_max + 1):

                #Do something real here
                #The caluclated value shoud be a float

                x = float(y)
                tf_matrix.append(x)

        self.gs = gs_values
        self.gs_targets = gs_indices

        self.tf_matrix = tf_matrix

    def get_source_and_target(self):

        return self.gs, self.gs_targets

    def get_transposed_im(self, im):

        return self.tf_matrix.take(im.astype(np.int))

    def get_transposed_im_old(self, im):

        im2 = np.zeros(im.shape) * np.nan
        tf = self.tf_matrix
        for i in xrange(tf.size):

            np.place(im2, im == i, tf[i])

        return im2
'''


class Image_Analysis():

    def __init__(self, path=None, image=None, pattern_image_path=None,
                 scale=1.0, resource_paths=None):

        self._path = path
        self._img = None
        self._pattern_img = None
        self._load_error = None
        self._transformed = False
        self._conversion_factor = 1.0 / scale
        self._logger = logging.getLogger("Resource Image Analysis")

        if os.path.isfile(pattern_image_path) is False and resource_paths is not None:

            pattern_image_path = os.path.join(resource_paths.images, os.path.basename(
                pattern_image_path))

        if pattern_image_path:

            try:

                pattern_img = plt.imread(pattern_image_path)

            except:

                self._logger.error(
                    "Could not open orientation guide image at " +
                    str(pattern_image_path))

                self._load_error = True

            if self._load_error is not True:

                if len(pattern_img.shape) > 2:

                    pattern_img = pattern_img[:, :, 0]

                self._pattern_img = pattern_img

        if image is not None:

            self._img = np.asarray(image)

        if path:

            if not(self._img is None):

                self._logger.warning(
                    "Won't load from path since actually submitted")

            else:

                try:

                    self._img = plt.imread(path)

                except:

                    self._logger.error("Could not open image at " + str(path))
                    self._load_error = True

        if self._load_error is not True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]

    def load_other_size(self, path=None, conversion_factor=1.0):

        self._conversion_factor = float(conversion_factor)

        if path is None:

            path = self._path

        self._logger.info("Loading image from {0}".format(path))

        try:

            self._img = plt.imread(path)

        except:

            self._logger.error("Could not reload image at " + str(path))
            self._load_error = True

        if self._load_error is not True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]

        if conversion_factor != 1.0:

            self._logger.info("Scaling to {0}".format(conversion_factor))
            self._img = Quick_Scale_To_im(conversion_factor)
            self._logger.info("Scaled")

    def get_hit_refined(self, hit, conv_img):

        #quarter_stencil = map(lambda x: x/8.0, stencil_size)
        #m_hit = conv_img[(max_coord[0] - quarter_stencil[0] or 0):\
            #(max_coord[0] + quarter_stencil[0] or conv_img.shape[0]),
            #(max_coord[1] - quarter_stencil[1] or 0):\
            #(max_coord[1] + quarter_stencil[1] or conv_img.shape[1])]

        #mg = np.mgrid[:m_hit.shape[0],:m_hit.shape[1]]
        #w_m_hit = m_hit*mg *m_hit.shape[0] * m_hit.shape[1] /\
            #float(m_hit.sum())
        #refinement = np.array((w_m_hit[0].mean(), w_m_hit[1].mean()))
        #m_hit_max = np.where(m_hit == m_hit.max())
        #print "HIT: {1}, REFINED HIT: {0}, VECTOR: {2}".format(
            #refinement,
            #(m_hit_max[0][0], m_hit_max[1][0]),
            #(refinement - np.array([m_hit_max[0][0], m_hit_max[1][0]])))

        #print "OLD POS", max_coord, " ",
        #max_coord = (refinement - np.array([m_hit_max[0][0],
            #m_hit_max[1][0]])) + max_coord
        #print "NEW POS", max_coord, "\n"

        return hit

    def get_convolution(self, threshold=127):

        t_img = (self._img > threshold).astype(np.int8) * 2 - 1
        marker = self._pattern_img

        if len(marker.shape) == 3:

            marker = marker[:, :, 0]

        t_mrk = (marker > 0) * 2 - 1

        return fftconvolve(t_img, t_mrk, mode='same')

    def get_best_location(self, conv_img, stencil_size, refine_hit=True):
        """This whas hidden and should be taken care of, is it needed"""

        max_coord = np.where(conv_img == conv_img.max())

        if len(max_coord[0]) == 0:

            return None, conv_img

        max_coord = np.array((max_coord[0][0], max_coord[1][0]))

        #Refining
        if refine_hit:
            max_coord = self.get_hit_refined(max_coord, conv_img)

        #Zeroing out hit
        half_stencil = map(lambda x: x / 2.0, stencil_size)

        d1_min = (max_coord[0] - half_stencil[0] > 0 and
                  max_coord[0] - half_stencil[0] or 0)

        d1_max = (max_coord[0] + half_stencil[0] < conv_img.shape[0]
                  and max_coord[0] + half_stencil[0] or conv_img.shape[0] - 1)

        d2_min = (max_coord[1] - half_stencil[1] > 0 and
                  max_coord[1] - half_stencil[1] or 0)

        d2_max = (max_coord[1] + half_stencil[1] < conv_img.shape[1]
                  and max_coord[1] + half_stencil[1] or conv_img.shape[1] - 1)

        conv_img[d1_min: d1_max, d2_min:d2_max] = conv_img.min() - 1

        return max_coord, conv_img

    def get_best_locations(self, conv_img, stencil_size, n, refine_hit=True):
        """This returns the best locations as a list of coordinates on the
        CURRENT IMAGE regardless of if it was scaled"""

        m_locations = []
        c_img = conv_img.copy()
        i = 0
        try:
            n = int(n)
        except:
            n = 3

        while i < n:

            m_loc, c_img = self.get_best_location(c_img, stencil_size,
                                                  refine_hit)

            m_locations.append(m_loc)

            i += 1

        return m_locations

    def find_pattern(self, markings=3, img_threshold=127):
        """This function returns the image positions as numpy arrays that
        are scaled to match the ORIGINAL IMAGE size"""

        if self.get_loaded():

            c1 = self.get_convolution(threshold=img_threshold)

            m1 = np.array(self.get_best_locations(
                c1, self._pattern_img.shape,
                markings, refine_hit=False)) * self._conversion_factor

            return m1[:, 1], m1[:, 0]

        else:

            return None, None

    def get_loaded(self):

        return (self._img is not None) and (self._load_error is not True)


class Analyse_Grayscale(object):

    ORTH_EDGE_T = 0.2
    ORTH_T1 = 0.15
    ORTH_T2 = 0.3
    GS_ROUGH_INTENSITY_T1 = (256 * 1 / 4)
    GS_ROUGH_INTENSITY_T2 = 125
    GS_ROUGH_INTENSITY_T3 = 170
    SPIKE_UP_T = 1.2
    SPIKE_BEST_TOLLERANCE = 0.05
    SAFETY_PADDING = 0.2
    SAFETY_COEFF = 0.5
    NEW_GS_ALG_L_DIFF_T = 0.03
    NEW_GS_ALG_L_DIFF_SPIKE_T = 0.3
    NEW_GS_ALG_SPIKES_FRACTION = 0.8
    NEW_SAFETY_PADDING = 0.2

    def __init__(self, target_type="Kodak", image=None, scale_factor=1.0):

        global GRAYSCALES
        self.grayscale_type = target_type

        for k in GRAYSCALES[target_type]:

            setattr(self, "_grayscale_{0}".format(k),
                    target_type in resource_grayscale.GRAYSCALE_SCALABLE and
                    GRAYSCALES[target_type][k] * scale_factor or
                    GRAYSCALES[target_type][k])

            #print "Set self._grayscale_{0}".format(k)

        self._logger = logging.getLogger("Analyse Grayscale")
        self._img = image
        #np.save("tmp_img.npy", image)

        #Variables from analysis
        self._grayscale_pos = None
        self._grayscale = None
        self._grayscale_X = None

        if image is not None:

            self._grayscale_pos, self._grayscale = self.get_grayscale()

            self._grayscale_X = self.get_grayscale_X(self._grayscale_pos)

    @property
    def image(self):

        return self._img

    def get_target_values(self):

        return self._grayscale_targets

    def get_source_values(self):

        return self._grayscale

    def get_sampling_positions(self):

        return self._orthMid, self._grayscale_pos

    def get_grayscale_X(self, grayscale_pos=None):

        if grayscale_pos is None:

            grayscale_pos = self._grayscale_pos

        if grayscale_pos is None:

            return None

        X = np.array(grayscale_pos)
        median_distance = np.median(X[1:] - X[: -1])

        self._grayscale_X = [0]

        pos = 0

        for i in range(1, len(grayscale_pos)):

            pos += int(round((grayscale_pos[i] - grayscale_pos[i - 1])
                             / median_distance))

            self._grayscale_X.append(pos)

        return self._grayscale_X

    def _get_ortho_trimmed(self, rect):

        A = (self._img[self._img.shape[0] / 2:, :] <
             self.GS_ROUGH_INTENSITY_T1).astype(int)

        orth_diff = -1

        kern = np.asarray([-1, 0, 1])
        Aorth = A.mean(axis=0)
        Aorth_edge = abs(fftconvolve(kern, Aorth, 'same'))
        Aorth_signals = Aorth_edge > self.ORTH_EDGE_T
        Aorth_positions = np.where(Aorth_signals)

        if len(Aorth_positions[0]) > 1:

            Aorth_pos_diff = abs(
                1 - (Aorth_positions[0][1:] - Aorth_positions[0][:-1]) /
                float(self._grayscale_width))

            rect[0][1] = Aorth_positions[0][Aorth_pos_diff.argmin()]
            rect[1][1] = Aorth_positions[0][Aorth_pos_diff.argmin() + 1]
            orth_diff = rect[1][1] - rect[0][1]

        if orth_diff == -1:

            #Orthagonal trim second try
            firstPass = True
            in_strip = False
            old_orth = None

            for i, orth in enumerate(A.mean(axis=0)):

                if firstPass:

                    firstPass = False

                else:

                    if abs(old_orth - orth) > self.ORTH_T1:

                        if in_strip is False:

                            if orth > self.ORTH_T2:

                                rect[0][1] = i
                                in_strip = True

                        else:

                            rect[1][1] = i
                            break

                old_orth = orth

            orth_diff = rect[1][1] - rect[0][1]

        #safety margin

        if orth_diff < self._grayscale_min_width:

            delta = abs(orth_diff - self._grayscale_min_width) / 2
            rect[0][1] -= delta
            rect[1][1] += delta

        self._mid_orth_strip = rect[0][1] + (rect[1][1] - rect[0][1]) / 2

        return rect

    def _get_para_trimmed(self, rect):

        i = (rect[1][0] - rect[0][0]) / 2.0

        #DEBUG PLOT
        #plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        #plt.show()
        #END DEBUG PLOT

        strip_values = self._img[rect[0][0]: rect[1][0],
                                 rect[0][1]: rect[1][1]].mean(axis=1)

        #GET HIGH VALUE SECTION
        A2 = strip_values > self.GS_ROUGH_INTENSITY_T2
        A2_edges = np.convolve(A2, np.array([-1, 1]), mode='same')
        A2_up = np.where(A2_edges == -1)[0]
        A2_down = np.where(A2_edges == 1)[0]

        box_need = self._grayscale_higher_than_half_width

        for i, v in enumerate(A2_up):

            if len(A2_down) >= i + 1:

                if A2_down[i] - v > box_need:

                    rect[0][0] = v
                    rect[1][0] = A2_down[i]

                    break

        #GET LOW VALUE SECTION
        A2 = strip_values < self.GS_ROUGH_INTENSITY_T3
        A2_edges = np.convolve(A2, np.array([-1, 1]), mode='same')
        A2_up = np.where(A2_edges == -1)[0]
        A2_down = np.where(A2_edges == 1)[0]

        box_need = self._grayscale_lower_than_half_width

        for i, v in enumerate(A2_up):

            if len(A2_down) >= i + 1:

                if A2_down[i] - v > box_need:

                    if rect[0][0] < v < rect[1][0]:

                        rect[1][0] = A2_down[i]
                        break

                    elif rect[0][0] < A2_down[i] < rect[1][0]:

                        rect[0][0] = v
                        break

        return rect

    def _get_start_rect(self):

        return [[0, 0], [self._img.shape[0], self._img.shape[1]]]

    def _get_clean_im_and_rect(self):

        #np.save("gs_example.npy", self._img)

        rect = self._get_start_rect()

        #print "START:", rect

        rect = self._get_ortho_trimmed(rect)

        #print "ORTHO:", rect

        rect = self._get_para_trimmed(rect)

        #print "PARA:", rect

        im = self._img[rect[0][0]: rect[1][0],
                       rect[0][1]: rect[1][1]]

        self._orthMid = (rect[1][1] + rect[0][1]) / 2.0

        return im, rect

    def get_grayscale(self, image=None):

        if image is not None:

            self._img = image

        if self._img is None or sum(self._img.shape) == 0:

            return None

        #DEBUG PLOT
        #plt.imshow(self._img)
        #plt.show()
        #DEBUG PLOT END

        self._logger.debug("GRAYSCALE ANALYSIS: Of shape {0}".format(
                           self._img.shape))

        im_slice, rect = self._get_clean_im_and_rect()

        #THE 1D SIGNAL ALONG THE GS
        strip_values = im_slice.mean(axis=1)
        #FOUND GS-SEGMENT DIFFERENCE TO EXPECTED SIZE
        expected_strip_size = float(self._grayscale_length *
                                    self._grayscale_sections)
        gs_l_diff = abs(1 - strip_values.size / expected_strip_size)

        #FINDING SPIKES
        kernel = [-1, 1]  # old [-1,2,-1]
        up_spikes = np.abs(np.convolve(strip_values, kernel,
                           "same")) > self.SPIKE_UP_T
        up_spikes = r_signal.get_center_of_spikes(up_spikes)

        gray_scale_pos = None

        if gs_l_diff < self.NEW_GS_ALG_L_DIFF_T:

            expected_spikes = (np.arange(1, self._grayscale_sections) *
                               self._grayscale_length)

            expected_offset = (expected_strip_size - strip_values.size) / 2.0

            expected_spikes += expected_offset

            observed_spikes = np.where(up_spikes)[0]

            pos_diffs = np.abs(np.subtract.outer(
                observed_spikes,
                expected_spikes)).argmin(axis=0)

            deltas = []
            for ei, oi in enumerate(pos_diffs):
                deltas.append(abs(expected_spikes[ei] - observed_spikes[oi]))
                if deltas[-1] > self._grayscale_length * self.NEW_GS_ALG_L_DIFF_SPIKE_T:
                    deltas[-1] = np.nan
            deltas = np.array(deltas)

            #IF GS-SECTION SEEMS TO BE RIGHT SIZE FOR THE WHOLE GS
            #THEN THE SECTIONING PROBABLY IS A GOOD ESTIMATE FOR THE GS
            #IF SPIKES MATCHES MOST OF THE EXPECTED EDGES
            if ((np.isfinite(deltas).sum() - np.isnan(deltas[0]) -
                    np.isnan(deltas[-1])) / float(self._grayscale_sections) >
                    self.NEW_GS_ALG_SPIKES_FRACTION):

                edges = []
                for di, oi in enumerate(pos_diffs):
                    if np.isfinite(deltas[di]):
                        edges.append(observed_spikes[oi])
                    else:
                        edges.append(np.nan)
                edges = np.array(edges, dtype=np.float)
                nan_edges = np.isnan(edges)
                fin_edges = np.isfinite(edges)
                X = np.arange(edges.size, dtype=np.float) + 1
                edges[nan_edges] = np.interp(X[nan_edges], X[fin_edges],
                                             edges[fin_edges],
                                             left=np.nan,
                                             right=np.nan)
                fin_edges = np.isfinite(edges)
                where_fin_edges = np.where(fin_edges)[0]

                #GET THE FREQ
                frequency = np.diff(edges[where_fin_edges[0]: where_fin_edges[-1]], 1)
                frequency = frequency[np.isfinite(frequency)].mean()

                #EXTENDED TO GET OUTSIDE EDGES
                edges = np.r_[[np.nan], edges, [np.nan]]
                fin_edges = np.isfinite(edges)
                where_fin_edges = np.where(fin_edges)[0]
                for i in range(where_fin_edges[0] - 1, -1, -1):
                    edges[i] = edges[i + 1] - frequency
                for i in range(where_fin_edges[-1] + 1, edges.size):
                    edges[i] = edges[i - 1] + frequency

                #EXTRACTING SECTION MIDPOINTS
                gray_scale_pos = np.interp(
                    np.arange(self._grayscale_sections) + 0.5,
                    np.arange(self._grayscale_sections + 1),
                    edges)

                self._logger.info("GRAYSCALE: Got signal with new method")

                #CHECKING OVERFLOWS
                if gray_scale_pos[0] - frequency * self.NEW_SAFETY_PADDING < 0:
                    gray_scale_pos += frequency
                if (gray_scale_pos[-1] + frequency * self.NEW_SAFETY_PADDING >
                        strip_values.size):
                    gray_scale_pos -= frequency

                #SETTING ABS POS REL TO WHOLE IM-SECTION
                gray_scale_pos += rect[0][0]

                val_orth = self._grayscale_width * self.NEW_SAFETY_PADDING
                val_para = frequency * self.NEW_SAFETY_PADDING

                #SETTING VALUE TOP
                top = self._mid_orth_strip - val_orth
                if top < 0:
                    top = 0

                #SETTING VALUE BOTTOM
                bottom = self._mid_orth_strip + val_orth + 1
                if bottom >= self._img.shape[1]:
                    bottom = self._img.shape[1] - 1

                gray_scale = []
                for pos in gray_scale_pos:

                    left = pos - val_para

                    if left < 0:
                        left = 0

                    right = pos + val_para

                    if right >= self._img.shape[0]:
                        right = self._img.shape[0] - 1

                    gray_scale.append(self._img[left: right, top: bottom].mean())

            else:

                self._logger.warning(
                    "GRAYSCALE: Too bad signal for new method, using fallback")

        if gray_scale_pos is None:

            best_spikes = r_signal.get_best_spikes(
                up_spikes,
                self._grayscale_length,
                tollerance=self.SPIKE_BEST_TOLLERANCE,
                require_both_sides=False)

            frequency = r_signal.get_perfect_frequency2(
                best_spikes, self._grayscale_length)

            #Sections + 1 because actually looking at edges to sections
            offset = r_signal.get_best_offset(
                self._grayscale_sections + 1,
                best_spikes, frequency=frequency)

            signal = r_signal.get_true_signal(
                self._img.shape[0],
                self._grayscale_sections + 1,
                up_spikes, frequency=frequency,
                offset=offset)

            if signal is None:

                self._logger.warning((
                    "GRAYSCALE, no signal detected for f={0} and"
                    " offset={1} in best_spikes={2} from spikes={3}").format(
                        frequency, offset, best_spikes, up_spikes))

                self._grayscale = None
                return None, None

            ###DEBUG CUT SECTION
            """
            from matplotlib import pyplot as plt
            plt.clf()
            plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]],
                cmap=plt.cm.Greys_r)
            plt.plot(signal, "*")
            plt.show()
            """
            ###DEBUG END

            if signal[0] - frequency * self.SAFETY_PADDING < 0:

                self._logger.warning(
                    "GRAYSCALE, the signal got adjusted one interval"
                    " due to lower bound overshoot")

                signal += frequency

            if signal[-1] + frequency * self.SAFETY_PADDING > strip_values.size:

                self._logger.warning(
                    "GRAYSCALE, the signal got adjusted one interval"
                    " due to upper bound overshoot")

                signal -= frequency

            gray_scale = []
            gray_scale_pos = []

            self.ortho_half_height = self._grayscale_width / \
                2.0 * self.SAFETY_COEFF

            #SETTING TOP
            top = self._mid_orth_strip - self.ortho_half_height
            if top < 0:
                top = 0

            #SETTING BOTTOM
            bottom = self._mid_orth_strip + self.ortho_half_height
            if bottom >= self._img.shape[1]:
                bottom = self._img.shape[1] - 1

            for pos in xrange(signal.size - 1):

                mid = signal[pos:pos + 2].mean() + rect[0][0]

                gray_scale_pos.append(mid)

                left = gray_scale_pos[-1] - 0.5 * frequency * self.SAFETY_COEFF

                if left < 0:
                    left = 0

                right = gray_scale_pos[-1] + 0.5 * frequency * self.SAFETY_COEFF

                if right >= self._img.shape[0]:
                    right = self._img.shape[0] - 1

                gray_scale.append(self._img[left: right, top: bottom].mean())

        self._grayscale_pos = gray_scale_pos
        self._grayscale = gray_scale

        #print "GS", gray_scale
        #print "GS POS", gray_scale_pos
        return gray_scale_pos, gray_scale
