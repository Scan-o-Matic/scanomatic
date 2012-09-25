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

import sys
import os
import types
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
from scipy.optimize import fsolve
import numpy as np
import logging
import matplotlib.pyplot as plt

#
# SCANNOMATIC DEPENDENCIES
#

import resource_signal as r_signal

#
# FUNCTIONS
#


def Quick_Scale_To(source_path, target_path, source_dpi=600, target_dpi=150):

    small_im = Quick_Scale_To_im(source_path, source_dpi=source_dpi,
        target_dpi=target_dpi)

    try:

        np.save(target_path, small_im)

    except:

        logging.error("Could not save scaled down image")

        return -1


def Quick_Scale_To_im(source_path, source_dpi=600, target_dpi=150,
    scale=None):

    try:

        im = plt.imread(source_path)

    except:

        logging.error("Could not open source")

        return -1

    if scale is None:
        scale = target_dpi / float(source_dpi)

    small_im = zoom(im, scale, order=1)
 
    return small_im


class Image_Transpose(object):

    def __init__(self, *args, **kwargs):

        #So that p3 won't fail
        self.gs_a = 0
        self.gs_b = 0
        self.gs_c = 1
        self.gs_d = 0

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

        if gs_values != None:

            #Create value - indices if not supplied
            if gs_indices == None:

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

        self.tf_matrix = tf_matrix

    def get_transposed_im(self, im):

        im2 = np.zeros(im.shape) * np.nan
        tf = self.tf_matrix
        for i in xrange(tf.size):

            np.place(im2, im==i, tf[i])

        return im2


class Image_Analysis():

    def __init__(self, path=None, image=None, pattern_image_path=None):

        self._path = path
        self._img = None
        self._pattern_img = None
        self._load_error = None
        self._transformed = False
        self._conversion_factor = 1.0

        if pattern_image_path:

            try:

                pattern_img = plt.imread(pattern_image_path)

            except:

                logging.error("Could not open orientation guide image at " +
                        str(pattern_image_path))

                self._load_error = True

            if self._load_error != True:

                if len(pattern_img.shape) > 2:

                    pattern_img = pattern_img[:, :, 0]

                self._pattern_img = pattern_img
                #self._pattern_img = np.ones((pattern_img.shape[0]+8,
                        #pattern_img.shape[1]+8))
                #self._pattern_img[4:self._pattern_img.shape[0]-4,4:
                        #self._pattern_img.shape[1]-4] = pattern_img

        if image is not None:


            self._img = np.asarray(image)

        if path:

            if not(self._img is None):

                logging.warning("Won't load from path since actually submitted")

            else:

                try:

                    self._img = plt.imread(path)

                except:

                    logging.error("Could not open image at " + str(path))
                    self._load_error = True


        if self._load_error != True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]


    def load_other_size(self, path=None, conversion_factor=1.0):

        self._conversion_factor = float(conversion_factor)

        if path is None:

            path = self._path

        try:

            self._img = plt.imread(path)

        except:

            logging.error("Could not reload image at " + str(path))
            self._load_error = True

        if self._load_error != True:

            if len(self._img.shape) > 2:

                self._img = self._img[:, :, 0]

        if conversion_factor != 1.0:

            self._img = Quick_Scale_To_im(conversion_factor)


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

    def get_convolution(self, threshold=130):

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

        d1_min = (max_coord[0] - half_stencil[0] > 0 and \
                max_coord[0] - half_stencil[0] or 0)

        d1_max = (max_coord[0] + half_stencil[0] < conv_img.shape[0] \
                and max_coord[0] + half_stencil[0] or conv_img.shape[0] - 1)

        d2_min = (max_coord[1] - half_stencil[1] > 0 and \
                max_coord[1] - half_stencil[1] or 0)

        d2_max = (max_coord[1] + half_stencil[1] < conv_img.shape[1] \
                and max_coord[1] + half_stencil[1] or conv_img.shape[1] - 1)

        conv_img[d1_min: d1_max, d2_min:d2_max] = \
                conv_img.min() - 1

        return max_coord, conv_img

    def get_best_locations(self, conv_img, stencil_size, n, refine_hit=True):

        m_locations = []
        c_img = conv_img.copy()

        while len(m_locations) < n:

            m_loc, c_img = self.get_best_location(c_img, stencil_size,
                        refine_hit)

            m_locations.append(m_loc)

        return m_locations

    def find_pattern(self, markings=3, img_threshold=130):

        if self.get_loaded():

            c1 = self.get_convolution(threshold=img_threshold)

            m1 = np.array(self.get_best_locations(c1, self._pattern_img.shape,
                markings, refine_hit=False))

            return m1[:, 1], m1[:, 0]

        else:

            return None, None

    def get_loaded(self):

        return (self._img != None) and (self._load_error != True)

    def get_subsection(self, section):

        if self.get_loaded() and section is not None:

            """
            if section[0][1] > section[1][1]:

                upper = section[1][1]
                lower = section[0][1]

            else:

                upper = section[0][1]
                lower = section[1][1]

            if section[0][0] > section[1][0]:

                left = section[1][0]
                right = section[0][0]

            else:

                left = section[0][0]
                right = section[1][0]
            """

            section = zip(*map(sorted, zip(*section)))

            try:

                subsection = self._img[
                    section[0][0]: section[1][0],
                    section[0][1]: section[1][1]]
                """
                    left * self._conversion_factor):
                    int(right * self._conversion_factor),
                    int(upper * self._conversion_factor):
                    int(lower * self._conversion_factor)]
                """
            except:

                subsection = None

            return subsection

        return None


class Analyse_Grayscale(object):

    def __init__(self, target_type="Kodak", image=None, scale_factor=1.0, dpi=600):

        self.grayscale_type = target_type
        self._grayscale_dict = {
            'Kodak': {
                'targets': [82, 78, 74, 70, 66, 62, 58, 54, 50, 46, 42,
                    38, 34, 30, 26, 22, 18, 14, 10, 6, 4, 2, 0],
                'width': 55,
                'sections': 23,
                'lower_than_half_width': 350,
                'higher_than_half_width': 150,
                'length': 28.57
                }
            }

        for k in self._grayscale_dict[target_type]:

            setattr(self, "_grayscale_{0}".format(k),
                self._grayscale_dict[target_type][k])

        self._img = image

        #Variables from analysis
        self._grayscale_pos = None
        self._grayscale = None
        self._grayscale_X = None

        if image != None:

            self._grayscale_pos, self._grayscale = self.get_grayscale(
                scale_factor=scale_factor, dpi=dpi)

            self._grayscale_X = self.get_grayscale_X(self._grayscale_pos)

    def get_target_values(self):

        return self._grayscale_targets

    def get_grayscale_X(self, grayscale_pos=None):

        if grayscale_pos == None:

            grayscale_pos = self._grayscale_pos

        if grayscale_pos == None:

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

    def _get_ortho_trimmed(self, rect, scale_factor):

        A = (self._img[self._img.shape[0] / 2:, :] < (256 * 1 / 4)).astype(int)

        orth_diff = -1

        kern = np.asarray([-1, 0, 1])
        Aorth = A.mean(0)
        Aorth_edge = abs(fftconvolve(kern, Aorth, 'same'))
        Aorth_edge_threshold = 0.2  # Aorth_edge.max() * 0.70
        Aorth_signals = Aorth_edge > Aorth_edge_threshold
        Aorth_positions = np.where(Aorth_signals)

        if len(Aorth_positions[0]) > 1:

            Aorth_pos_diff = abs(1 - ((Aorth_positions[0][1:] - \
                Aorth_positions[0][:-1]) / float(self._grayscale_width)))
            rect[0][1] = Aorth_positions[0][Aorth_pos_diff.argmin()]
            rect[1][1] = Aorth_positions[0][Aorth_pos_diff.argmin() + 1]
            orth_diff = rect[1][1] - rect[0][1]

        if orth_diff == -1:

            #Orthagonal trim second try
            firstPass = True
            in_strip = False
            threshold = 0.15
            threshold2 = 0.3

            for i, orth in enumerate(A.mean(0)):

                if firstPass:

                    firstPass = False

                else:

                    if abs(old_orth - orth) > threshold:

                        if in_strip == False:

                            if orth > threshold2:

                                rect[0][1] = i
                                in_strip = True

                        else:

                            rect[1][1] = i
                            break

                old_orth = orth

            orth_diff = rect[1][1] - rect[0][1]

        #safty margin
        min_orths = 30 / scale_factor

        if orth_diff > min_orths:

            rect[0][1] += (orth_diff - min_orths) / 2
            rect[1][1] -= (orth_diff - min_orths) / 2

        self._mid_orth_strip = rect[0][1] + (rect[1][1] - rect[0][1]) / 2

        return rect

    def _get_para_trimmed(self, rect, scale_factor):


        i = (rect[1][0] - rect[0][0]) / 2.0

        #DEBUG PLOT
        #plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]])
        #plt.show()
        #END DEBUG PLOT

        strip_values = self._img[rect[0][0]: rect[1][0],
                            rect[0][1]: rect[1][1]].mean(1)

        #GET HIGH VALUE SECTION
        A2 = strip_values > 125
        A2_edges = np.convolve(A2, np.array([-1, 1]), mode='same')
        A2_up = np.where(A2_edges == -1)[0]
        A2_down = np.where(A2_edges == 1)[0]

        box_need = self._grayscale_higher_than_half_width / \
                                                scale_factor

        for i, v in enumerate(A2_up):

            if len(A2_down) >= i+1:

                if A2_down[i] - v > box_need:

                    rect[0][0] =  v
                    rect[1][0] = A2_down[i]

                    break

        #GET LOW VALUE SECTION
        A2 = strip_values < 170
        A2_edges = np.convolve(A2, np.array([-1, 1]), mode='same')
        A2_up = np.where(A2_edges == -1)[0]
        A2_down = np.where(A2_edges == 1)[0]

        box_need = self._grayscale_lower_than_half_width / \
                                 scale_factor

        for i, v in enumerate(A2_up):

            if len(A2_down) >= i+1:

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

    def get_grayscale(self, image=None, scale_factor=1.0, dpi=600):

        if image != None:

            self._img = image

        if self._img is None or sum(self._img.shape) == 0:

            return None

        #DEBUG PLOT
        #plt.imshow(self._img)
        #plt.show()
        #DEBUG PLOT END

        if scale_factor == 1 and dpi != 600:

            scale_factor = 600.0 / dpi

        logging.debug("GRAYSCALE ANALYSIS: Of images "
            "{0} at dpi={1} and scale_factor={2}".format(
            self._img.shape, dpi, scale_factor))
        
        rect = self._get_start_rect()

        rect = self._get_ortho_trimmed(rect, scale_factor)

        rect = self._get_para_trimmed(rect, scale_factor)

        strip_values = self._img[rect[0][0]: rect[1][0],
                        rect[0][1]: rect[1][1]].mean(1)

        threshold = 1.2
        kernel = [-1, 1]  # old [-1,2,-1]

        up_spikes = np.abs(np.convolve(strip_values, kernel,
                "same")) > threshold

        up_spikes = r_signal.get_center_of_spikes(up_spikes)

        best_spikes = r_signal.get_best_spikes(up_spikes,
            self._grayscale_length / scale_factor,
            tollerance=0.05,
            require_both_sides=False)

        frequency = r_signal.get_perfect_frequency2(best_spikes,
            self._grayscale_length / scale_factor)

        #Sections + 1 because actually looking at edges to sections
        offset = r_signal.get_best_offset(
            self._grayscale_sections + 1,
            best_spikes, frequency=frequency)

        signal = r_signal.get_true_signal(self._img.shape[0],
            self._grayscale_sections + 1,
            up_spikes, frequency=frequency,
            offset=offset)

        if signal is None:

            logging.warning(("GRAYSCALE, no signal detected for f={0} and"
                " offset={1} in best_spikes={2} from spikes={3}").format(
                frequency, offset, best_spikes, up_spikes))

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

        safety_buffer = 0.2

        if signal[0] + frequency * safety_buffer < 0:

            logging.warning("GRAYSCALE, the signal got adjusted one interval"
                " due to lower bound overshoot")

            signal += frequency

        if signal[-1] - frequency * safety_buffer > strip_values.size:

            logging.warning("GRAYSCALE, the signal got adjusted one interval"
                " due to upper bound overshoot")

            signal -= frequency

        safety_coeff = 0.5
        gray_scale = []
        gray_scale_pos = []

        self.ortho_half_height = self._grayscale_width / \
            2.0 * safety_coeff

        top = self._mid_orth_strip - self.ortho_half_height
        bottom = self._mid_orth_strip + self.ortho_half_height

        for pos in xrange(signal.size - 1):

            mid = signal[pos:pos + 2].mean() + rect[0][0]

            gray_scale_pos.append(mid)

            left = gray_scale_pos[-1] - 0.5 * frequency * safety_coeff
            right = gray_scale_pos[-1] + 0.5 * frequency * safety_coeff

            gray_scale.append(self._img[left: right, top: bottom].mean())

        self._gray_scale_pos = gray_scale_pos
        self._gray_scale = gray_scale

        return gray_scale_pos, gray_scale
