import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d
import os

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
from grayscale import getGrayscale
import signal
from scanomatic.generics.maths import mid50_mean as iqr_mean
from scanomatic.io.paths import Paths

#
# GOBALS
#

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

_logger = logger.Logger("Analyze Grayscale")

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
    detection = np.abs(convolve2d(im_scaled, kernel_scaled, mode="valid"))
    peak = gaussian_filter1d(np.max(detection, axis=0), half_width).argmax()

    return im[:, int(round(peak - half_width)): int(round(peak + half_width))]


def get_para_trimmed_slice(im_ortho_trimmed, grayscale, kernel_part_of_segment=0.6, permissibility_threshold=20,
                           acceptability_threshold=0.8, padding=0.7):

    # Restructures the image so that local variances can be measured using a kernel the scaled (default 0.7) size
    # of the segment size

    kernel_size = tuple(int(kernel_part_of_segment * v) for v in (grayscale['length'], grayscale['width']))

    try:
        strided_im = as_strided(im_ortho_trimmed,
                                shape=(im_ortho_trimmed.shape[0] - kernel_size[0] + 1,
                                       im_ortho_trimmed.shape[1] - kernel_size[1] + 1,
                                       kernel_size[0], kernel_size[1]),
                                strides=im_ortho_trimmed.strides * 2)
    except ValueError:
        _logger.error("Failed to stride image, try making a larger selection around the grayscale on the fixture.")
        return None

    # Note: ortho_signal has indices half kernel_size offset with regards to im_ortho_trimmed

    ortho_signal = np.median(np.var(strided_im, axis=(-1, -2)), axis=1) / sum(kernel_size)

    # Possibly more sophisticated method may be needed looking at establishing the drifting baseline and convolving
    # segment-lengths with one-kernels to ensure no peak is there.

    permissible_positions = ortho_signal < permissibility_threshold
    """:type : numpy.ndarray"""
    # Selects the best stretch of permissible signal (True) compared to the expected length of the grayscale
    acceptable_placement = None
    placement_accuracy = 0
    in_section = False
    section_start = 0
    length = float(grayscale['sections'] * grayscale['length'])

    for i, val in enumerate(permissible_positions):

        if in_section and not val:

            in_section = False
            # The difference of the observed length compared to the exepected is divided with the expected.
            # It is practically impossible due to restraints on size of area checked for grayscale that the
            # delta is larger than the expected length. For that reason the division will be in the range 0 - 1
            # with better precision being close to 0. Accuracy will therefore be close to 1 if the fit is good.
            accuracy = 1 - abs(i - section_start - length) / length
            if accuracy > placement_accuracy:
                placement_accuracy = accuracy
                acceptable_placement_length = i - 1 - section_start
                acceptable_placement = int(acceptable_placement_length / 2) + section_start
        elif not in_section and val:
            in_section = True
            section_start = i

    if in_section:
        # This only repeats above code in the loop but covers the case that the segment ends with a
        # permissible area.
        accuracy = 1 - abs(permissible_positions.size - section_start - length) / length
        if accuracy > placement_accuracy:
            placement_accuracy = accuracy
            acceptable_placement_length = permissible_positions.size - 1 - section_start
            acceptable_placement = int(acceptable_placement_length / 2) + section_start

    if placement_accuracy > acceptability_threshold:

        # Using the expected length of the grayscale (which implies that this has to be a good value
        # buffering is scaled by the accuracy of the selected segments length compare to the expected
        # length.
        buffered_half_length = int(round(length / 2 + grayscale['length'] * padding * (1 - placement_accuracy)))

        # Correct offset in the permissible signal to the image
        acceptable_placement += kernel_size[0] / 2

        return im_ortho_trimmed[int(round(max(0, acceptable_placement - buffered_half_length))):
                                int(round(min(im_ortho_trimmed.shape[0],
                                              acceptable_placement + buffered_half_length)))]

    return im_ortho_trimmed


def get_grayscale(fixture, grayscale_area_model, debug=False):
    im = fixture.get_grayscale_im_section(grayscale_area_model)
    if im is None:
        return None
    return get_grayscale_image_analysis(im, grayscale_area_model.name, debug=debug)


def get_grayscale_image_analysis(im, grayscale_name, debug=False):
    global DEBUG_DETECTION

    gs = getGrayscale(grayscale_name)
    if not im.size:
        return None, None
    im_o = get_ortho_trimmed_slice(im, gs)
    if not im_o.size:
        return None, None
    im_p = get_para_trimmed_slice(im_o, gs)
    if im_p is None or not im_p.size:
        return None, None
    DEBUG_DETECTION = debug

    return detect_grayscale(im_p, gs)


def is_valid_grayscale(calibration_target_values, image_values, pixel_depth=8):

    try:
        fit = np.polyfit(image_values, calibration_target_values, 3)
    except TypeError:
        # Probably vectors were of unequal size
        _logger.error("Probable mismatch between number of detected segments and expected number of segments.")
        return False

    poly = np.poly1d(fit)
    data = poly(np.arange(2*pixel_depth))

    # Analytical derivative over the value span ensuring that the log2_curve is continuously increasing or decreasing
    poly_is_ok = np.unique(np.sign(data[1:] - data[:-1])).size == 1
    if not poly_is_ok:
        _logger.warning("Polynomial fit failed required monotonous test")

    # Verify that the same sign correlation is intact for the difference of two consecutive elements in each series
    measures_are_ok = np.unique(tuple(np.sign(a) - np.sign(b) for a, b in
                                      zip(np.diff(np.convolve(calibration_target_values, [1, 1, 1], 'valid')),
                                          np.diff(np.convolve(image_values, [1, 1, 1], 'valid'))))).size == 1

    if not measures_are_ok:
        _logger.warning("Actual measures lack monotonous tendencies.")
    return poly_is_ok and measures_are_ok


def detect_grayscale(im_trimmed, grayscale):

    gray_scale = []
    grayscale_segment_centers = []

    if im_trimmed is None or sum(im_trimmed.shape) == 0:

        _logger.error("No image loaded or null image")
        return None, None

    rect = ([0, 0], im_trimmed.shape)
    mid_ortho_slice = (rect[1][1] + rect[0][1]) / 2.0
    mid_ortho_trimmed = mid_ortho_slice - rect[0][1]
    _logger.info("Loaded pre-trimmed image slice for GS detection")

    if DEBUG_DETECTION:
        np.save(os.path.join(Paths().log, 'gs_section_used_in_detection.npy'), im_trimmed)

    # THE 1D SIGNAL ALONG THE GS
    para_signal_trimmed_im = np.mean(im_trimmed, axis=1)

    if DEBUG_DETECTION:
        np.save(os.path.join(Paths().log, 'gs_para_signal_trimmed_im.npy'), para_signal_trimmed_im)

    # FOUND GS-SEGMENT DIFFERENCE TO EXPECTED SIZE
    expected_strip_size = float(grayscale['length'] * grayscale['sections'])

    gs_l_diff = abs(1 - para_signal_trimmed_im.size / expected_strip_size)

    up_spikes = signal.get_signal(para_signal_trimmed_im, SPIKE_UP_T)

    if DEBUG_DETECTION:
        np.save(os.path.join(Paths().log, "gs_up_spikes.npy"), up_spikes)

    if gs_l_diff < NEW_GS_ALG_L_DIFF_T:

        _logger.info('Using default grayscale detection method')

        deltas, observed_spikes, observed_to_expected_map = signal.get_signal_data(
            para_signal_trimmed_im, up_spikes, grayscale,
            grayscale["length"] * NEW_GS_ALG_L_DIFF_SPIKE_T)

        # IF GS-SECTION SEEMS TO BE RIGHT SIZE FOR THE WHOLE GS
        # THEN THE SECTIONING PROBABLY IS A GOOD ESTIMATE FOR THE GS
        # IF SPIKES MATCHES MOST OF THE EXPECTED EDGES
        if ((np.isfinite(deltas).sum() - np.isnan(deltas[0]) -
                np.isnan(deltas[-1])) / float(grayscale['sections']) >
                NEW_GS_ALG_SPIKES_FRACTION):

            if DEBUG_DETECTION:
                np.save(os.path.join(Paths().log, "gs_pos_diffs.npy"), observed_to_expected_map)
                np.save(os.path.join(Paths().log, "gs_deltas.npy"), deltas)
                np.save(os.path.join(Paths().log, "gs_observed_spikes.npy"), observed_spikes)

            edges = signal.get_signal_edges(observed_to_expected_map, deltas, observed_spikes,
                                            grayscale['sections'])

            fin_edges = np.isfinite(edges)
            if not fin_edges.any():
                _logger.error("No finite edges found")
                return None, None

            where_fin_edges = np.where(fin_edges)[0]

            if DEBUG_DETECTION:
                np.save(os.path.join(Paths().log, "gs_edges.npy"), edges)

            # GET THE FREQ
            frequency = np.diff(edges[where_fin_edges[0]: where_fin_edges[-1]], 1)
            frequency = frequency[np.isfinite(frequency)].mean()

            if not np.isfinite(frequency):
                _logger.error("No frequency was detected, thus no grayscale")
                return None, None

            edges = signal.extrapolate_edges(edges, frequency, para_signal_trimmed_im.size)

            if edges.size != grayscale['sections'] + 1:
                _logger.error(
                    "Number of edges doesn't correspond to the grayscale segments ({0}!={1})".format(
                        edges.size, grayscale['sections'] + 1))
                return None, None

            # EXTRACTING SECTION MIDPOINTS
            grayscale_segment_centers = np.interp(
                np.arange(grayscale['sections']) + 0.5,
                np.arange(grayscale['sections'] + 1),
                edges)

            _logger.info("GRAYSCALE: Got signal with new method")

            # CHECKING OVERFLOWS
            if grayscale_segment_centers[0] - frequency * NEW_SAFETY_PADDING < 0:
                grayscale_segment_centers += frequency
            if (grayscale_segment_centers[-1] + frequency * NEW_SAFETY_PADDING >
                    para_signal_trimmed_im.size):
                grayscale_segment_centers -= frequency

            # SETTING ABS POS REL TO WHOLE IM-SECTION
            grayscale_segment_centers += rect[0][0]
            _logger.info("Offsetting centers with {0}".format(rect[0][0]))

            if DEBUG_DETECTION:
                np.save(os.path.join(Paths().log, "gs_segment_centers.npy"), grayscale_segment_centers)

            val_orth = grayscale['width'] * NEW_SAFETY_PADDING
            val_para = frequency * NEW_SAFETY_PADDING

            # SETTING VALUE TOP
            top = mid_ortho_trimmed - val_orth
            if top < 0:
                top = 0

            # SETTING VALUE BOTTOM
            bottom = mid_ortho_trimmed + val_orth + 1
            if bottom >= im_trimmed.shape[1]:
                bottom = im_trimmed.shape[1] - 1

            if DEBUG_DETECTION:
                np.save(os.path.join(Paths().log, "gs_slice.npy"), im_trimmed)

            for i, pos in enumerate(grayscale_segment_centers):

                left = pos - val_para

                if left < 0:
                    left = 0

                right = pos + val_para

                if right >= im_trimmed.shape[0]:
                    right = im_trimmed.shape[0] - 1

                left = int(round(left))
                right = int(round(right))
                top = int(round(top))
                bottom = int(round(bottom))

                gray_scale.append(iqr_mean(im_trimmed[left: right, top: bottom]))

                if DEBUG_DETECTION:
                    np.save(os.path.join(Paths().log, "gs_segment_{0}.npy".format(i)),
                            im_trimmed[left: right, top: bottom])

        else:

            _logger.warning("New method failed, using fallback")

    else:

        _logger.warning("Skipped new method, threshold not met ({0} > {1}; slice {2})".format(
            gs_l_diff, NEW_GS_ALG_L_DIFF_T, rect))

    if len(grayscale_segment_centers) == 0:

        _logger.warning("Using fallback method")

        best_spikes = signal.get_best_spikes(
            up_spikes,
            grayscale['length'],
            tollerance=SPIKE_BEST_TOLLERANCE,
            require_both_sides=False)

        frequency = signal.get_perfect_frequency2(
            best_spikes, grayscale['length'])

        # Sections + 1 because actually looking at edges to sections
        offset = signal.get_best_offset(
            grayscale['sections'] + 1,
            best_spikes, frequency=frequency)

        s = signal.get_true_signal(
            im_trimmed.shape[0],
            grayscale['sections'] + 1,
            up_spikes, frequency=frequency,
            offset=offset)

        if s is None:

            _logger.warning((
                "GRAYSCALE, no signal detected for f={0} and"
                " offset={1} in best_spikes={2} from spikes={3}").format(
                    frequency, offset, best_spikes, up_spikes))

            return None, None

        if s[0] - frequency * SAFETY_PADDING < 0:

            _logger.warning(
                "GRAYSCALE, the signal got adjusted one interval"
                " due to lower bound overshoot")

            s += frequency

        if s[-1] + frequency * SAFETY_PADDING > para_signal_trimmed_im.size:

            _logger.warning(
                "GRAYSCALE, the signal got adjusted one interval"
                " due to upper bound overshoot")

            s -= frequency

        ortho_half_height = grayscale['width'] / 2.0 * SAFETY_COEFF

        # SETTING TOP
        top = mid_ortho_trimmed - ortho_half_height
        if top < 0:
            top = 0

        # SETTING BOTTOM
        bottom = mid_ortho_trimmed + ortho_half_height
        if bottom >= im_trimmed.shape[1]:
            bottom = im_trimmed.shape[1] - 1

        for pos in xrange(s.size - 1):

            mid = s[pos:pos + 2].mean() + rect[0][0]

            grayscale_segment_centers.append(mid)

            left = grayscale_segment_centers[-1] - 0.5 * frequency * SAFETY_COEFF

            if left < 0:
                left = 0

            right = grayscale_segment_centers[-1] + 0.5 * frequency * SAFETY_COEFF

            if right >= im_trimmed.shape[0]:
                right = im_trimmed.shape[0] - 1

            gray_scale.append(iqr_mean(im_trimmed[left: right, top: bottom]))

    gray_scale, grayscale_segment_centers = signal.get_higher_second_half_order_according_to_first(
        gray_scale, grayscale_segment_centers)

    if DEBUG_DETECTION:
        np.save(os.path.join(Paths().log, "gs_final_values.npy"), gray_scale)

    return grayscale_segment_centers, gray_scale
