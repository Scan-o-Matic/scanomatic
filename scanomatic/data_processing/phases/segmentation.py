import operator
from itertools import izip

import numpy as np
from enum import Enum
from scipy import signal
from scipy.ndimage import label, generic_filter


class CurvePhases(Enum):
    """Phases of curves recognized

    Attributes:
        CurvePhases.Multiple: Several types for same position, to be
            considered as an error.
        CurvePhases.Undetermined: Positions yet to be classified
            or not fulfilling any classification
        CurvePhases.Flat: Positions that exhibit no growth or collapse
        CurvePhases.GrowthAcceleration: Positions that are
            characterized by a positive second derivative
            and positive derivative.
        CurvePhases.GrowthRetardation: Positions that are
            characterized by a negative second derivative
            and positive derivative.
        CurvePhases.Impulse: Close to linear segment with growth.
        CurvePhases.Collapse: Close to linear segment with decreasing
            population size.
        CurvePhases.CollapseAcceleration: Positions that are
            characterized by a positive second derivative
            and negative derivative.
        CurvePhases.CollapseRetardation: Positions that are
            characterized by a negative second derivative
            and negative derivative.
        CurvePhases.UndeterminedNonLinear: Positions of curves that
            have only been determined not to be linear.
        CurvePhases.UndeterminedNonFlat: Positions that are not flat
            but whose properties otherwise has yet to be determined

    """
    Multiple = -1
    """:type : CurvePhases"""
    Undetermined = 0
    """:type : CurvePhases"""
    Flat = 1
    """:type : CurvePhases"""
    GrowthAcceleration = 2
    """:type : CurvePhases"""
    GrowthRetardation = 3
    """:type : CurvePhases"""
    Impulse = 4
    """:type : CurvePhases"""
    Collapse = 5
    """:type : CurvePhases"""
    CollapseAcceleration = 6
    """:type : CurvePhases"""
    CollapseRetardation = 7
    """:type : CurvePhases"""
    UndeterminedNonLinear = 8
    """:type : CurvePhases"""
    UndeterminedNonFlat = 9
    """:type : CurvePhases"""


class Thresholds(Enum):
    """Thresholds used by the phase algorithm

    Attributes:
        Thresholds.LinearModelExtension:
            Factor for impulse and collapse slopes to be
            considered equal to max/min point.
        Threshold.PhaseMinimumLength:
            The number of measurements needed for a segment to be
            considered detected.
        Thresholds.FlatlineSlopRequirement:
            Maximum slope for something to be flatline.
        Thresholds.UniformityThreshold:
            The fraction of positions considered that must agree on a
            certain direction of the first or second derivative.
        Thresholds.UniformityTestMinSize:
            The number of measurements included in the
            `UniformityThreshold` test.
        Thresholds.NonFlatLinearMinimumLength:
            Minimum length of collapse or impulse

    """
    LinearModelExtension = 0
    """:type : Thresholds"""
    PhaseMinimumLength = 1
    """:type : Thresholds"""
    FlatlineSlopRequirement = 2
    """:type : Thresholds"""
    UniformityThreshold = 3
    """:type : Thresholds"""
    UniformityTestMinSize = 4
    """:type : Thresholds"""
    SecondDerivativeSigmaAsNotZero = 5
    """:type : Thresholds"""
    NonFlatLinearMinimumLength = 7
    """:type : Thresholds"""


class PhaseEdge(Enum):
    """Segment edges

    Attributes:
        PhaseEdge.Left: Left edge
        PhaseEdge.Right: Right edge
        PhaseEdge.Intelligent: Most interesting edge
    """
    Left = 0
    """:type : PhaseEdge"""
    Right = 1
    """:type : PhaseEdge"""
    Intelligent = 2
    """:type : PhaseEdge"""


DEFAULT_THRESHOLDS = {
    Thresholds.LinearModelExtension: 0.01,
    Thresholds.PhaseMinimumLength: 3,
    Thresholds.NonFlatLinearMinimumLength: 7,
    Thresholds.FlatlineSlopRequirement: 0.02,
    Thresholds.UniformityThreshold: 0.4,
    Thresholds.UniformityTestMinSize: 7,
    Thresholds.SecondDerivativeSigmaAsNotZero: 0.5}


def segment(times, curve, dydt, dydt_signs_flat, ddydt_signs, phases, offset, thresholds=None):
    """Iteratively segments a curve into its component CurvePhases

    Proposed future segmentation structure:

        mark everything as flat segments or non-flat

        for each non-flat and not non-linear segment:
            if contains linear slope:
                mark slope as impulse or collapse
                for each flanking:
                    detetect non-linear type
                    if nothing detected, mark as linear
            else
                mark as non-linear

        for each remaining non-linear segment:
            if contains detectable non-linear type:
                mark type
            else:
                mark undefined

    Args:
        times:
            The sample times vector
        curve:
            The smooth growth curve
        dydt:
            The first derivative
        dydt_signs_flat:
            The signs of first derivative with area around 0
            considered 0.
        ddydt_signs:
            The signs of the second derivative with area around 0
            considered 0.
        phases:
            The phase classification vector (should be all 0)
        offset:
            An int for offset between curve and derivative
        thresholds:
            The thresholds dictionary to be used.
    """
    curve = np.ma.masked_invalid(np.log2(curve))

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Mark all flats
    _set_flat_segments(dydt_signs_flat,
                       thresholds[Thresholds.PhaseMinimumLength],
                       phases)

    yield None

    while (phases == CurvePhases.UndeterminedNonFlat.value).any():

        # Mark linear slope
        flanking = _set_nonflat_linear_segment(
            times,
            curve,
            dydt,
            dydt_signs_flat,
            thresholds[Thresholds.LinearModelExtension],
            thresholds[Thresholds.NonFlatLinearMinimumLength],
            offset, phases)

        yield None

        if flanking.any():

            first_on_left_flank = flanking.argmin()

            for filt in _get_candidate_segment(flanking):

                direction = PhaseEdge.Right if \
                    filt.argmin() == first_on_left_flank else \
                    PhaseEdge.Left

                # Mark flanking non-linear phase
                phase = _set_nonlinear_phase_type(
                    dydt, dydt_signs_flat, ddydt_signs, filt,
                    direction,
                    thresholds[Thresholds.UniformityTestMinSize],
                    thresholds[Thresholds.UniformityThreshold],
                    thresholds[Thresholds.PhaseMinimumLength],
                    offset, phases)

                if phase is CurvePhases.Undetermined:
                    # If no curved segment found, it is not safe to look for more
                    # non-flat linear phases because could merge two that should
                    # not be merged.
                    phases[filt] = CurvePhases.UndeterminedNonLinear.value

                # Only look for the first non-linear segment rest is up for grabs for
                # Next iteration of finding impulses or collapses
                flanking[filt] = False

                yield None

    # Try to classify remaining positions as non linear phases
    for filt in _get_candidate_segment(phases, test_value=CurvePhases.UndeterminedNonLinear.value):

        phase = _set_nonlinear_phase_type(
            dydt, dydt_signs_flat, ddydt_signs, filt,
            PhaseEdge.Intelligent,
            thresholds[Thresholds.UniformityTestMinSize],
            thresholds[Thresholds.UniformityThreshold],
            thresholds[Thresholds.PhaseMinimumLength],
            offset, phases)

        yield None

        # If currently considered segment had no phase then it is undetermined
        if phase is CurvePhases.Undetermined:

            phases[filt] = phase.value
            yield None

    # If there's an offset assume phase carries to edge
    if offset:
        phases[:offset] = phases[offset]
        phases[-offset:] = phases[-offset - 1]
        yield None

    # Bridge neighbouring segments of same type if gap is one
    _bridge_gaps(phases)


def _bridge_gaps(phases):
    """Fills in undefined gaps if same phase on each side

    Maximum gap size is 1

    :param phases: The phase classification array
    """

    undefined, = np.where(phases == CurvePhases.Undetermined.value)
    last_index = phases.size - 1

    # If the curve is just two measurements this makes little sense
    if last_index < 2:
        return

    for loc in undefined:

        if loc == 0:
            if phases[1] != CurvePhases.Undetermined.value:
                phases[loc] = phases[loc + 1]
        elif loc == last_index:
            if phases[loc - 1] != CurvePhases.Undetermined.value:
                phases[loc] = phases[loc - 1]
        elif phases[loc - 1] == phases[loc + 1] and phases[loc + 1] != CurvePhases.Undetermined.value:
            phases[loc] = phases[loc + 1]


def _set_flat_segments(dydt_signs, minimum_segmentlength, phases):

    phases[...] = CurvePhases.UndeterminedNonFlat.value
    flats = _bridge_canditates(dydt_signs == 0)
    for length, left, right in izip(*_get_candidate_lengths_and_edges(flats)):
        if length >= minimum_segmentlength:
            phases[left: right] = CurvePhases.Flat.value


def _get_candidate_segment(complex_segment, test_value=True):
    """While complex_segment contains any test_value the first
    segment of such will be returned as a boolean array

    :param complex_segment: an array
    :param test_value: the value to look for
    :return: generator
    """
    while True:
        labels, n_labels = label(complex_segment == test_value)

        if n_labels:
            yield labels == 1
        else:
            break


def _set_nonflat_linear_segment(times, curve, dydt, dydt_signs, extension_threshold,
                                minimum_length_threshold, offset, phases):

    # All positions with sufficient slope
    filt = phases == CurvePhases.UndeterminedNonFlat.value

    # In case there are small regions left
    if not filt.any():

        phases[phases == CurvePhases.UndeterminedNonFlat.value] = CurvePhases.UndeterminedNonLinear.value
        # Since no segment was detected there are no bordering segments
        return np.array([])

    # Determine value and position of steepest slope
    loc_slope = np.abs(dydt[filt]).max()
    loc = np.where((np.abs(dydt) == loc_slope) & filt)[0][0]

    # Getting back the sign and values for linear model
    loc_slope = dydt[loc]
    loc_value = curve[loc]
    loc_time = times[loc]

    # Tangent at max
    tangent = (times - loc_time) * loc_slope + loc_value

    # Determine comparison operator for first derivative
    phase = CurvePhases.Collapse if loc_slope < 0 else CurvePhases.Impulse

    # Find all candidates
    candidates = (np.abs(curve - tangent) < extension_threshold * loc_value).filled(False)
    candidates &= filt
    candidates = _bridge_canditates(candidates)
    candidates, n_found = label(candidates)

    # Verify that there's actually still a candidate at the peak value
    if n_found == 0:

        phases[filt] = CurvePhases.UndeterminedNonLinear.value

        # Since no segment was detected there are no bordering segments
        return np.array([])

    # Get the true phase positions from the candidates
    elected = candidates == candidates[loc]

    # Verify that the elected phase fulfills length threshold
    if elected.sum() < minimum_length_threshold:

        phases[elected] = CurvePhases.UndeterminedNonLinear.value
        # Since no segment was detected there are no bordering segments
        return np.array([])

    # Update filt for border detection below before updating elected!
    filt = (phases == CurvePhases.Undetermined.value) | \
           (phases == CurvePhases.UndeterminedNonLinear.value) | \
           (phases == CurvePhases.UndeterminedNonFlat.value)

    # Only consider flanking those that have valid sign.
    # TODO: Note that it can cause an issue if curve is very wack, could merge two segments that shouldn't be
    # Probably extremely unlikely
    op1 = operator.le if phase is CurvePhases.Collapse else operator.ge
    filt &= op1(dydt_signs, 0)

    # Set the detected phase
    if offset:
        phases[offset: -offset][elected] = phase.value
    else:
        phases[elected] = phase.value

    # Locate flanking segments
    border_candidates, _ = label(filt)
    loc_label = border_candidates[loc]
    return (border_candidates == loc_label) - elected


def _get_candidate_lengths_and_edges(candidates):

    kernel = [-1, 1]
    edges = signal.convolve(candidates, kernel, mode='same')
    lefts, = np.where(edges == -1)
    rights, = np.where(edges == 1)
    if rights.size < lefts.size:
        rights = np.hstack((rights, candidates.size))

    return rights - lefts, lefts, rights


def _bridge_canditates(candidates, window_size=5):
    # TODO: Verify method, use published, sure this will never expand initial detections?
    for window in range(3, window_size, 2):
        candidates = signal.medfilt(candidates, window_size).astype(bool) | candidates
    return candidates


def _set_nonlinear_phase_type(dydt, dydt_signs, ddydt_signs, filt, test_edge, test_min_length,
                              uniformity_threshold, min_length, offset, phases):
    """ Determines type of non-linear phase.

    Function filters the first and second derivatives, only looking
    at a number of measurements near one of the two edges of the
    candidate region. The signs of each (1st and 2nd derivative)
    are used to determine the type of phase.

    Note:
        Both derivatives need a sufficient deviation from 0 to be
        considered to have a sign.

    Args:
        dydt: The slope values
        dydt_signs: The sing of the first derivative
        ddydt_signs: The sign of the second derivative
        filt: Boolean array of positions considered
        test_edge: At which edge (left or right) of the filt the
            test should be performed
        uniformity_threshold: The degree of conformity in sign needed
            I.e. the fraction of ddydt_signs in the test that must
            point in the same direction. Or the fraction of
            dydt_signs that have to do the same.
        test_min_length: How many points should be tested as a minimum
        min_length: Minimum length to be considered a detected phase
        offset: Offset of first derivative values to curve
        phases: The phase-classification array

    Returns: The phase type, any of the following
        CurvePhases.Undetermined (failed detection),
        CurvePhases.GrowthAcceleration,
        CurvePhases.CollapseAcceleration,
        CurvePhases.GrowthRetardation,
        CurvePhases.CollapseRetardation

    """
    phase = CurvePhases.Undetermined

    # Define type at one of the edges
    if test_edge is PhaseEdge.Intelligent:

        # This takes a rough estimate of which side is more interesting
        # based on the location of the steepest slope

        phase = _classify_non_linear_segment(dydt_signs, ddydt_signs, uniformity_threshold)
        if phase == CurvePhases.Undetermined:
            steepest_loc = np.abs(dydt[filt]).argmax()
            test_edge = PhaseEdge.Left if steepest_loc / float(filt.sum()) < 0.5 else PhaseEdge.Right

    if test_edge is PhaseEdge.Left:
        for test_length in range(test_min_length, dydt.size, 4):
            ddydt_section = ddydt_signs[filt][:test_length]
            dydt_section = dydt_signs[filt][:test_length]
            phase = _classify_non_linear_segment(dydt_section, ddydt_section, uniformity_threshold)
            if phase != CurvePhases.Undetermined:
                break
    elif test_edge is PhaseEdge.Right:
        for test_length in range(test_min_length, dydt.size, 4):
            ddydt_section = ddydt_signs[filt][-test_length:]
            dydt_section = dydt_signs[filt][-test_length:]
            phase = _classify_non_linear_segment(dydt_section, ddydt_section, uniformity_threshold)
            if phase != CurvePhases.Undetermined:
                break

    elif phase == CurvePhases.Undetermined:
        return CurvePhases.Undetermined

    # Determine which operators to be used for first (op1) and second (op2) derivative signs
    if phase is CurvePhases.GrowthAcceleration or phase is CurvePhases.GrowthRetardation:
        op1 = operator.ge
    else:
        op1 = operator.le

    if phase is CurvePhases.GrowthAcceleration or phase is CurvePhases.CollapseRetardation:
        op2 = operator.ge
    else:
        op2 = operator.le

    candidates = filt & op1(dydt_signs, 0) & op2(ddydt_signs, 0)
    candidates = generic_filter(candidates, _custom_filt, size=9, mode='nearest')
    candidates, label_count = label(candidates)

    if label_count:

        candidates = candidates == (1 if test_edge is PhaseEdge.Left else label_count)

        if candidates.sum() < min_length:
            return CurvePhases.Undetermined

        if offset:
            phases[offset: -offset][candidates] = phase.value
        else:
            phases[candidates] = phase.value

        return phase

    else:

        return CurvePhases.Undetermined


def _classify_non_linear_segment(dydt, ddydt, uniformity_threshold):
    """Classifies non linear segment

    Args:
        dydt: First derivative signs
        ddydt: Second derivative signs
        uniformity_threshold:

    Returns: CurvePhase

    """

    if ddydt.size == 0 or ddydt.sum() == 0 or dydt.sum() == 0:
        return CurvePhases.Undetermined

    # Classify as acceleration or retardation
    sign = np.sign(ddydt.mean())
    if sign == 0:
        return CurvePhases.Undetermined
    op = operator.le if sign < 0 else operator.ge
    value = op(ddydt, 0).mean() * sign

    if value > uniformity_threshold:
        candidate_phase_types = (CurvePhases.GrowthAcceleration, CurvePhases.CollapseRetardation)
    elif value < -uniformity_threshold:
        candidate_phase_types = (CurvePhases.GrowthRetardation, CurvePhases.CollapseAcceleration)
    else:
        return CurvePhases.Undetermined

    # Classify as acceleration or retardation
    sign = np.sign(dydt.mean())
    if sign == 0:
        return CurvePhases.Undetermined
    op = operator.le if sign < 0 else operator.ge
    value = op(dydt, 0).mean() * sign

    if value > uniformity_threshold:
        return candidate_phase_types[0]
    elif value < -uniformity_threshold:
        return candidate_phase_types[1]
    else:
        return CurvePhases.Undetermined


def _custom_filt(v, max_gap=3, min_length=3):

    w, = np.where(v)
    if not w.any():
        return False
    filted = signal.convolve(v[w[0]:w[-1] + 1] == np.False_, (1,)*max_gap, mode='same') < max_gap
    padded = np.hstack([(0,), filted, (0,)]).astype(int)
    diff = np.diff(padded)
    return (np.where(diff < 0)[0] - np.where(diff > 0)[0]).max() >= min_length


def get_data_needed_for_segmentation(phenotyper_object, plate, pos, threshold_for_sign, threshold_flatline):

    curve = phenotyper_object.smooth_growth_data[plate][pos]

    # Smoothing kernel for derivatives
    gauss = signal.gaussian(7, 3)
    gauss /= gauss.sum()

    # Some center weighted smoothing of derivative, we only care for general shape
    dydt = signal.convolve(phenotyper_object.get_derivative(plate, pos), gauss, mode='valid')
    d_offset = (phenotyper_object.times.size - dydt.size) / 2
    dydt = np.hstack(([dydt[0] for _ in range(d_offset)], dydt, [dydt[-1] for _ in range(d_offset)]))

    dydt_ranks = np.abs(dydt).argsort().argsort()
    offset = (phenotyper_object.times.shape[0] - dydt.shape[0]) / 2

    # Smoothing in kernel shape because only want reliable trends
    ddydt = signal.convolve(dydt, [1, 0, -1], mode='valid')
    ddydt = signal.convolve(ddydt, gauss, mode='valid')

    dd_offset = (dydt.size - ddydt.size) / 2
    ddydt = np.hstack(([ddydt[0] for _ in range(dd_offset)], ddydt, [ddydt[-1] for _ in range(dd_offset)]))
    phases = np.ones_like(curve).astype(np.int) * CurvePhases.Undetermined.value
    """:type : numpy.ndarray"""

    # Determine second derviative signs
    ddydt_signs = np.sign(ddydt)
    ddydt_signs[np.abs(ddydt) < threshold_for_sign * ddydt[np.isfinite(ddydt)].std()] = 0

    # Determine first derivative signs for flattness questions
    dydt_signs_flat = np.sign(dydt)
    dydt_signs_flat[np.abs(dydt) < threshold_flatline] = 0

    return dydt, dydt_ranks, dydt_signs_flat, ddydt, ddydt_signs, phases, offset, curve