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
from numpy.lib.stride_tricks import as_strided
from scipy.signal import fftconvolve, convolve2d, convolve
from scipy.ndimage import gaussian_filter1d
import os

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import grayscale
import signal
from scanomatic.generics.maths import iqr_mean
from scanomatic.io.paths import Paths

#
# CLASSES
#


def get_ortho_trimmed_slice(im, grayscale):
    half_width = grayscale['width'] / 2
    im_scaled = im / float(im.max()) - 0.5
    kernel = np.array(grayscale['targets']).repeat(grayscale['length'])
    kernel = kernel.reshape((kernel.size, 1))
    if kernel.size > im.shape[0]:
        return np.array([])

    kernel_scaled = kernel / float(kernel.max()) - 0.5
    C = np.abs(convolve2d(im_scaled, kernel_scaled, mode="valid"))
    peak = gaussian_filter1d(np.max(C, axis=0), half_width).argmax()

    return im[:, peak - half_width: peak + half_width]


def get_para_trimmed_slice(im_ortho_trimmed, grayscale, kernel_part_of_segment=0.6, permissibility_threshold=20,
                           acceptability_threshold=0.8, buffer=0.5):

    # Restructures the image so that local variances can be measured using a kernel the scaled (default 0.7) size
    # of the segment size

    kernel_size = tuple(int(kernel_part_of_segment * v) for v in (grayscale['length'], grayscale['width']))
    strided_im = as_strided(im_ortho_trimmed,
                            shape=(im_ortho_trimmed.shape[0] - kernel_size[0] + 1,
                                   im_ortho_trimmed.shape[1] - kernel_size[1] + 1,
                                   kernel_size[0], kernel_size[1]),
                            strides=im_ortho_trimmed.strides * 2)

    # Note: ortho_signal has indices half kernel_size offset with regards to im_ortho_trimmed

    ortho_signal = np.median(np.var(strided_im, axis=(-1, -2)), axis=1) / sum(kernel_size)

    # Possibly more sophisticated method may be needed looking at establishing the drifting baseline and convolving
    # segment-lengths with one-kernels to ensure no peak is there.

    permissible_positions = ortho_signal < permissibility_threshold


    # Selects the best stretch of permissible signal (True) compared to the expected length of the grayscale
    acceptable_placement = None
    placement_accuracy = 0
    in_section = False
    section_start = 0
    length = float(grayscale['sections'] * grayscale['length'])

    for i, val in enumerate(permissible_positions):

        if in_section and not val:

            in_section = False
            accuracy = 1 - abs(i - section_start - length) / length
            if accuracy > placement_accuracy:
                placement_accuracy = accuracy
                acceptable_placement = int((i - 1 - section_start) / 2) + section_start
        elif not in_section and val:
            in_section = True
            section_start = i

    if in_section:
        accuracy = 1 - abs(i - section_start - length) / length
        if accuracy > placement_accuracy:
            placement_accuracy = accuracy
            acceptable_placement = int((permissible_positions.size - 1 - section_start) / 2) + section_start

    print (placement_accuracy)
    if placement_accuracy > acceptability_threshold:

        buffered_half_length = int(round(length / 2 + grayscale['length'] * buffer))

        # Correct offset in the permissible signa to the image
        acceptable_placement += kernel_size[0] / 2

        return im_ortho_trimmed[max(0, acceptable_placement - buffered_half_length):
                                min(im_ortho_trimmed.shape[0], acceptable_placement + buffered_half_length)]

    return im_ortho_trimmed


def get_grayscale(fixture, grayscale_area_model, debug=False):

    gs = grayscale.getGrayscale(grayscale_area_model.name)
    im = fixture.get_grayscale_im_section(grayscale_area_model)
    if not im.size:
        return None, None
    im_o = get_ortho_trimmed_slice(im, gs)
    if not im_o.size:
        return None, None
    im_p = get_para_trimmed_slice(im_o, gs)
    if not im_p.size:
        return None, None
    Analyse_Grayscale.DEBUG_DETECTION = debug
    ag = Analyse_Grayscale(target_type=grayscale_area_model.name, image=None, scale_factor=1)
    return ag.get_grayscale(im_p, pre_trimmed=True)


def is_valid_grayscale(calibration_target_values, image_values, pixel_depth=8) :

    try:
        fit = np.polyfit(image_values, calibration_target_values, 3)
    except TypeError:
        # Probably vectors were of unequal size
        return False

    poly = np.poly1d(fit)
    data = poly(np.arange(2*pixel_depth))

    # Analytical derivative over the value span ensuring that the curve is continuously increasing or decreasing
    poly_is_ok = np.unique(np.sign(data[1:] - data[:-1])).size == 1

    # Verify that the same sign correlation is intact for the difference of two consequtive elements in each series
    measures_are_ok = np.unique(tuple(np.sign(a) - np.sign(b) for a, b in
                                      zip(np.diff(calibration_target_values), np.diff(image_values)))).size == 1

    return poly_is_ok and measures_are_ok


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
    NEW_GS_ALG_L_DIFF_T = 0.1
    NEW_GS_ALG_L_DIFF_SPIKE_T = 0.3
    NEW_GS_ALG_SPIKES_FRACTION = 0.8
    NEW_SAFETY_PADDING = 0.2
    DEBUG_DETECTION = False

    def __init__(self, target_type="Kodak", image=None, scale_factor=1.0):

        self.grayscale_type = target_type
        self._grayscale = grayscale.getGrayscale(target_type)
        for k, v in self._grayscale.items():

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

        return self._mid_ortho_slice, self._grayscale_pos

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

        self._mid_ortho_trimmed = rect[0][1] + (rect[1][1] - rect[0][1]) / 2

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

        if self.DEBUG_DETECTION:

            np.save(os.path.join(Paths().log, "gs_slice.npy"), self._img)

        rect_in_slice_coords = self._get_start_rect()

        if self.DEBUG_DETECTION:
            self._logger.info("start rect is: {0}".format(rect_in_slice_coords))

        rect_in_slice_coords = self._get_ortho_trimmed(rect_in_slice_coords)

        if self.DEBUG_DETECTION:
            self._logger.info("Ortho trimmed rect is: {0}".format(rect_in_slice_coords))
            np.save(os.path.join(Paths().log, "gs_ortho_trimmed.npy"),
                    self._img[rect_in_slice_coords[0][0]: rect_in_slice_coords[1][0],
                    rect_in_slice_coords[0][1]: rect_in_slice_coords[1][1]])

        rect_in_slice_coords = self._get_para_trimmed(rect_in_slice_coords)

        if self.DEBUG_DETECTION:
            self._logger.info("Ortho & para trimmed rect is: {0}".format(rect_in_slice_coords))
            np.save(os.path.join(Paths().log, "gs_ortho_and_para_trimmed.npy"),
                    self._img[rect_in_slice_coords[0][0]: rect_in_slice_coords[1][0],
                    rect_in_slice_coords[0][1]: rect_in_slice_coords[1][1]])

        im_trimmed = self._img[rect_in_slice_coords[0][0]: rect_in_slice_coords[1][0],
                     rect_in_slice_coords[0][1]: rect_in_slice_coords[1][1]]

        self._mid_ortho_slice = (rect_in_slice_coords[1][1] + rect_in_slice_coords[0][1]) / 2.0

        return im_trimmed, rect_in_slice_coords

    def get_grayscale(self, image=None, pre_trimmed=False):

        self._sectionAreaSlices = []

        if image is not None:

            self._img = image

        if pre_trimmed:
            im_trimmed = self._img
            rect = ([0,0], self._img.shape)
            self._mid_ortho_slice = (rect[1][1] + rect[0][1]) / 2.0
            self._mid_ortho_trimmed = self._mid_ortho_slice - rect[0][1]
            self._logger.info("Loaded pre-trimmed image slice")

        else:

            im_trimmed, rect = self._get_clean_im_and_rect()
            self._logger.info("Using automatic trimming of image slice")

        if self._img is None or sum(self._img.shape) == 0:

            self._logger.error("No image loaded or null image")
            return None

        if self.DEBUG_DETECTION:
            np.save(os.path.join(Paths().log, 'gs_section_used_in_detection.npy'), im_trimmed)

        # THE 1D SIGNAL ALONG THE GS
        para_signal_trimmed_im = np.mean(im_trimmed, axis=1)

        if self.DEBUG_DETECTION:
            np.save(os.path.join(Paths().log, 'gs_para_signal_trimmed_im.npy'), para_signal_trimmed_im)

        # FOUND GS-SEGMENT DIFFERENCE TO EXPECTED SIZE
        expected_strip_size = float(self._grayscale_length *
                                    self._grayscale_sections)

        gs_l_diff = abs(1 - para_signal_trimmed_im.size / expected_strip_size)

        up_spikes = signal.get_signal(para_signal_trimmed_im, self.SPIKE_UP_T)
        grayscale_segment_centers = None
        if self.DEBUG_DETECTION:
            np.save(os.path.join(Paths().log, "gs_up_spikes.npy"), up_spikes)

        if gs_l_diff < Analyse_Grayscale.NEW_GS_ALG_L_DIFF_T:

            deltas, observed_spikes, observed_to_expected_map = signal.get_signal_data(
                para_signal_trimmed_im, up_spikes, self._grayscale,
                self._grayscale["length"] * Analyse_Grayscale.NEW_GS_ALG_L_DIFF_SPIKE_T)

            # IF GS-SECTION SEEMS TO BE RIGHT SIZE FOR THE WHOLE GS
            # THEN THE SECTIONING PROBABLY IS A GOOD ESTIMATE FOR THE GS
            # IF SPIKES MATCHES MOST OF THE EXPECTED EDGES
            if ((np.isfinite(deltas).sum() - np.isnan(deltas[0]) -
                    np.isnan(deltas[-1])) / float(self._grayscale_sections) >
                    self.NEW_GS_ALG_SPIKES_FRACTION):

                if self.DEBUG_DETECTION:
                    np.save(os.path.join(Paths().log, "gs_pos_diffs.npy"), observed_to_expected_map)
                    np.save(os.path.join(Paths().log, "gs_deltas.npy"), deltas)
                    np.save(os.path.join(Paths().log, "gs_observed_spikes.npy"), observed_spikes)

                edges = signal.get_signal_edges(observed_to_expected_map, deltas, observed_spikes,
                                                self._grayscale_sections)

                fin_edges = np.isfinite(edges)
                where_fin_edges = np.where(fin_edges)[0]

                if self.DEBUG_DETECTION:
                    np.save(os.path.join(Paths().log, "gs_edges.npy"), edges)

                # GET THE FREQ
                frequency = np.diff(edges[where_fin_edges[0]: where_fin_edges[-1]], 1)
                frequency = frequency[np.isfinite(frequency)].mean()

                edges = signal.extrapolate_edges(edges, frequency, para_signal_trimmed_im.size)

                if edges.size != self._grayscale_sections + 1:
                    self._logger.critical(
                        "Number of edges doesn't correspond to the grayscale segments ({0}!={1})".format(
                            edges.size, self._grayscale_sections + 1))
                    return None, None

                # EXTRACTING SECTION MIDPOINTS
                grayscale_segment_centers = np.interp(
                    np.arange(self._grayscale_sections) + 0.5,
                    np.arange(self._grayscale_sections + 1),
                    edges)

                self._logger.info("GRAYSCALE: Got signal with new method")

                #CHECKING OVERFLOWS
                if grayscale_segment_centers[0] - frequency * self.NEW_SAFETY_PADDING < 0:
                    grayscale_segment_centers += frequency
                if (grayscale_segment_centers[-1] + frequency * self.NEW_SAFETY_PADDING >
                        para_signal_trimmed_im.size):
                    grayscale_segment_centers -= frequency

                #SETTING ABS POS REL TO WHOLE IM-SECTION
                grayscale_segment_centers += rect[0][0]

                val_orth = self._grayscale_width * self.NEW_SAFETY_PADDING
                val_para = frequency * self.NEW_SAFETY_PADDING

                #SETTING VALUE TOP
                top = self._mid_ortho_trimmed - val_orth
                if top < 0:
                    top = 0

                #SETTING VALUE BOTTOM
                bottom = self._mid_ortho_trimmed + val_orth + 1
                if bottom >= self._img.shape[1]:
                    bottom = self._img.shape[1] - 1

                gray_scale = []
                if self.DEBUG_DETECTION:
                    np.save(os.path.join(Paths().log, "gs_slice.npy"), self._img)

                for i, pos in enumerate(grayscale_segment_centers):

                    left = pos - val_para

                    if left < 0:
                        left = 0

                    right = pos + val_para

                    if right >= self._img.shape[0]:
                        right = self._img.shape[0] - 1

                    self._sectionAreaSlices.append((slice(left, right),
                                                    slice(top, bottom)))

                    gray_scale.append(iqr_mean(self._img[left: right, top: bottom]))

                    if self.DEBUG_DETECTION:
                        np.save(os.path.join(Paths().log, "gs_segment_{0}.npy".format(i)),
                                self._img[left: right, top: bottom])

            else:

                self._logger.warning("New method failed, using fallback")

        else:

            self._logger.warning("Skipped new method, threshold not met ({0} > {1}; slice {2})".format(
                gs_l_diff, Analyse_Grayscale.NEW_GS_ALG_L_DIFF_T, rect))

        if grayscale_segment_centers is None:

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

            if s[-1] + frequency * self.SAFETY_PADDING > para_signal_trimmed_im.size:

                self._logger.warning(
                    "GRAYSCALE, the signal got adjusted one interval"
                    " due to upper bound overshoot")

                s -= frequency

            gray_scale = []
            grayscale_segment_centers = []

            self.ortho_half_height = self._grayscale_width / \
                2.0 * self.SAFETY_COEFF

            #SETTING TOP
            top = self._mid_ortho_trimmed - self.ortho_half_height
            if top < 0:
                top = 0

            #SETTING BOTTOM
            bottom = self._mid_ortho_trimmed + self.ortho_half_height
            if bottom >= self._img.shape[1]:
                bottom = self._img.shape[1] - 1

            for pos in xrange(s.size - 1):

                mid = s[pos:pos + 2].mean() + rect[0][0]

                grayscale_segment_centers.append(mid)

                left = grayscale_segment_centers[-1] - 0.5 * frequency * self.SAFETY_COEFF

                if left < 0:
                    left = 0

                right = grayscale_segment_centers[-1] + 0.5 * frequency * self.SAFETY_COEFF

                if right >= self._img.shape[0]:
                    right = self._img.shape[0] - 1

                gray_scale.append(iqr_mean(self._img[left: right, top: bottom]))

        self._grayscale_pos = grayscale_segment_centers
        self._grayscaleSource = gray_scale

        #print "GS", gray_scale
        #print "GS POS", gray_scale_pos
        gray_scale, grayscale_segment_centers = signal.get_higher_second_half_order_according_to_first(
            gray_scale, grayscale_segment_centers)

        return grayscale_segment_centers, gray_scale