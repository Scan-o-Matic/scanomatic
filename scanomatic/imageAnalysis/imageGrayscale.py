"""Detects grayscales in images"""

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

import numpy as np
from scipy.signal import fftconvolve

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import grayscale
import signal

#
# CLASSES
#


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

        for k, v in grayscale.getGrayscale(target_type).items():

            setattr(self, "_grayscale_{0}".format(k),
                    k in grayscale.GRAYSCALE_SCALABLE and
                    v * scale_factor or v)

        self._logger = logger.Logger("Analyze Grayscale")
        self._img = image
        #np.save("tmp_img.npy", image)

        #Variables from analysis
        self._grayscale_pos = None
        self._grayscaleSource = None
        self._grayscale_X = None
        self._sectionAreaSlices = []

        if image is not None:

            self._grayscale_pos, self._grayscaleSource = self.get_grayscale()

            self._grayscale_X = self.get_grayscale_X(self._grayscale_pos)

        else:
            self._logger.warning("No analysis run yet")

    @property
    def image(self):

        return self._img

    @property
    def slices(self):
        return [self._img[s] for s in self._sectionAreaSlices]

    def get_target_values(self):

        return self._grayscale_targets

    def get_source_values(self):

        return self._grayscaleSource

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

        self._sectionAreaSlices = []

        if image is not None:

            self._img = image

        if self._img is None or sum(self._img.shape) == 0:

            self._logger.error("No image loaded or null image")
            return None

        #DEBUG PLOT
        #plt.imshow(self._img)
        #plt.show()
        #DEBUG PLOT END

        self._logger.debug("Image shape {0}".format(self._img.shape))

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
        up_spikes = signal.get_center_of_spikes(up_spikes)

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

                    self._sectionAreaSlices.append((slice(left, right),
                                                    slice(top, bottom)))

                    gray_scale.append(self._img[left: right, top: bottom].mean())

            else:

                self._logger.warning("New method failed, using fallback")

        else:

            self._logger.warning("Skipped new method, threshold not met")

        if gray_scale_pos is None:

            self._logger.warning("Using fallback method")

            best_spikes = signal.get_best_spikes(
                up_spikes,
                self._grayscale_length,
                tollerance=self.SPIKE_BEST_TOLLERANCE,
                require_both_sides=False)

            frequency = signal.get_perfect_frequency2(
                best_spikes, self._grayscale_length)

            #Sections + 1 because actually looking at edges to sections
            offset = signal.get_best_offset(
                self._grayscale_sections + 1,
                best_spikes, frequency=frequency)

            s = signal.get_true_signal(
                self._img.shape[0],
                self._grayscale_sections + 1,
                up_spikes, frequency=frequency,
                offset=offset)

            if s is None:

                self._logger.warning((
                    "GRAYSCALE, no signal detected for f={0} and"
                    " offset={1} in best_spikes={2} from spikes={3}").format(
                        frequency, offset, best_spikes, up_spikes))

                self._grayscaleSource = None
                return None, None

            ###DEBUG CUT SECTION
            """
            from matplotlib import pyplot as plt
            plt.clf()
            plt.imshow(self._img[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]],
                cmap=plt.cm.Greys_r)
            plt.plot(s, "*")
            plt.show()
            """
            ###DEBUG END

            if s[0] - frequency * self.SAFETY_PADDING < 0:

                self._logger.warning(
                    "GRAYSCALE, the signal got adjusted one interval"
                    " due to lower bound overshoot")

                s += frequency

            if s[-1] + frequency * self.SAFETY_PADDING > strip_values.size:

                self._logger.warning(
                    "GRAYSCALE, the signal got adjusted one interval"
                    " due to upper bound overshoot")

                s -= frequency

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

            for pos in xrange(s.size - 1):

                mid = s[pos:pos + 2].mean() + rect[0][0]

                gray_scale_pos.append(mid)

                left = gray_scale_pos[-1] - 0.5 * frequency * self.SAFETY_COEFF

                if left < 0:
                    left = 0

                right = gray_scale_pos[-1] + 0.5 * frequency * self.SAFETY_COEFF

                if right >= self._img.shape[0]:
                    right = self._img.shape[0] - 1

                gray_scale.append(self._img[left: right, top: bottom].mean())

        self._grayscale_pos = gray_scale_pos
        self._grayscaleSource = gray_scale

        #print "GS", gray_scale
        #print "GS POS", gray_scale_pos
        return gray_scale_pos, gray_scale
