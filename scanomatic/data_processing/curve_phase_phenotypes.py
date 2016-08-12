import numpy as np
import operator
from scipy import signal
from scipy.ndimage import label, generic_filter
from scipy.stats import linregress
from collections import deque
from enum import Enum
from itertools import izip

from scanomatic.data_processing import growth_phenotypes

# TODO: Image segments_phasing_bug_0_11_35.png shows unexpected phases
# Need detection of linear phases based on prolonged linearity
# Need linear phase detection to run to completion until no linear phases remain


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
        Threshold.LinearModelMinimumLength:
            The number of measurements needed for a linear segment
            to be considered detected.
        Thresholds.FlatlineSlopRequirement:
            Maximum slope for something to be flatline.
        Thresholds.UniformityThreshold:
            The fraction of positions considered that must agree on a
            certain direction of the first or second derivative.
        Thresholds.UniformityTestSize:
            The number of measurements included in the
            `UniformityThreshold` test.

    """
    LinearModelExtension = 0
    """:type : Thresholds"""
    LinearModelMinimumLength = 1
    """:type : Thresholds"""
    FlatlineSlopRequirement = 2
    """:type : Thresholds"""
    UniformityThreshold = 3
    """:type : Thresholds"""
    UniformityTestSize = 4
    """:type : Thresholds"""
    SecondDerivativeSigmaAsNotZero = 5
    """:type : Thresholds"""


class CurvePhasePhenotypes(Enum):
    """Phenotypes for individual curve phases.

    _NOTE_ Some apply only to some `CurvePhases`.

    Attributes:
        CurvePhasePhenotypes.PopulationDoublingTime: The average population doubling time of the segment
        CurvePhasePhenotypes.Duration: The length of the segment in time.
        CurvePhasePhenotypes.FractionYield: The proportion of population doublings for the entire experiment
            that this segment is responsible for
        CurvePhasePhenotypes.Start: Start time of the segment
        CurvePhasePhenotypes.LinearModelSlope: The slope of the linear model fitted to the segment
        CurvePhasePhenotypes.LinearModelIntercept: The intercept of the linear model fitted to the segment
        CurvePhasePhenotypes.AsymptoteAngle: The angle between the initial point slope and the final point slope
            of the segment
        CurvePhasePhenotypes.AsymptoteIntersection: The intercept between the asymptotes as fraction of the `Duration`
            of the segment.
    """

    PopulationDoublingTime = 1
    """type: CurvePhasePhenotypes"""
    Duration = 2
    """type: CurvePhasePhenotypes"""
    FractionYield = 3
    """type: CurvePhasePhenotypes"""
    Start = 4
    """type: CurvePhasePhenotypes"""
    LinearModelSlope = 5
    """type: CurvePhasePhenotypes"""
    LinearModelIntercept = 6
    """type: CurvePhasePhenotypes"""
    AsymptoteAngle = 7
    """type: CurvePhasePhenotypes"""
    AsymptoteIntersection = 8
    """type: CurvePhasePhenotypes"""


class CurvePhaseMetaPhenotypes(Enum):
    """Phenotypes of an entire growth-curve based on the phase segmentation.

    Attributes:
        CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution:
            The fraction of the total yield (in population doublings) that the
            `CurvePhases.Impulse` that contribute most to the total yield is
            responsible for (`CurvePhasePhenotypes.FractionYield`).
        CurvePhaseMetaPhenotypes.FirstMinorImpulseYieldContribution:
            As with `CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution`
            but for the second most important `CurvePhases.Impulse`
        CurvePhaseMetaPhenotypes.MajorImpulseAveragePopulationDoublingTime:
            The `CurvePhases.Impulse` that contribute most to the
            total yield, its average population doubling time
            (`CurvePhasePhenotypes.PopulationDoublingTime`).
        CurvePhaseMetaPhenotypes.FirstMinorImpulseAveragePopulationDoublingTime:
            The average population doubling time of
            the second most contributing `CurvePhases.Impulse`

        CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteAngle:
            The `CurvePhasePhenotypes.AsymptoteAngle` of the first `CurvePhases.Acceleration`
        CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteAngle:
            The `CurvePhasePhenotypes.AsymptoteAngle` of the last `CurvePhases.Retardation`
        CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteIntersect:
            The `CurvePhasePhenotypes.AsymptoteIntersection` of the first `CurvePhases.Acceleration`
        CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteIntersect:
            The `CurvePhasePhenotypes.AsymptoteIntersection` of the last `CurvePhases.Retardation`

        CurvePhaseMetaPhenotypes.InitialLag:
            The intercept time of the linear model of the first `CurvePhases.Flat` and the first
            `CurvePhases.Impulse`. Note that this does not have to be the major impulse in the above
            measurements.
        CurvePhaseMetaPhenotypes.ExperimentDoublings:
            (Not implemented) Total doublings
        CurvePhaseMetaPhenotypes.Modalities:
            The number of `CurvePhases.Impulse`
        CurvePhaseMetaPhenotypes.Collapses:
            The number of `CurvePhases.Collapse`

        CurvePhaseMetaPhenotypes.ResidualGrowth:
            (Not implemented) Classifying the growth that happens after the last `CurvePhases.Impulse`.

    See Also:
        filter_plate: Get one of these out of a plate of phase segmentation information
    """
    MajorImpulseYieldContribution = 0
    FirstMinorImpulseYieldContribution = 1
    MajorImpulseAveragePopulationDoublingTime = 5
    FirstMinorImpulseAveragePopulationDoublingTime = 6

    InitialAccelerationAsymptoteAngle = 10
    FinalRetardationAsymptoteAngle = 11
    InitialAccelerationAsymptoteIntersect = 15
    FinalRetardationAsymptoteIntersect = 16

    InitialLag = 20
    ExperimentDoublings = 21
    InitialLagAlternativeModel = 22

    Modalities = 25
    Collapses = 26

    ResidualGrowth = 30


class VectorPhenotypes(Enum):
    """The vector type phenotypes used to store phase segmentation

    Attributes:
        VectorPhenotypes.PhasesClassifications:
            1D vector the same length as growth data with the `CurvePhases` values
            for classification of which phase each population size measurement in the growth data
            is classified as.
        VectorPhenotypes.PhasesPhenotypes:
            1D vector of `CurvePhasePhenotypes` keyed dicts for each segment in the curve.
    """
    PhasesClassifications = 0
    """:type : VectorPhenotypes"""
    PhasesPhenotypes = 1
    """:type : VectorPhenotypes"""


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


def _filter_find(vector, filt, func=np.max):
    vector = np.abs(vector)
    return np.where((vector == func(vector[filt])) & filt)[0]

DEFAULT_THRESHOLDS = {
    Thresholds.LinearModelExtension: 0.75,
    Thresholds.LinearModelMinimumLength: 3,
    Thresholds.FlatlineSlopRequirement: 0.02,
    Thresholds.UniformityThreshold: 0.66,
    Thresholds.UniformityTestSize: 7,
    Thresholds.SecondDerivativeSigmaAsNotZero: 0.5}


def _segment(dydt, dydt_signs, ddydt_signs, phases, filt, offset, thresholds=None):
    """Iteratively segments a curve into its component CurvePhases

    Proposed future segmentation structure:

        mark everything as flat segments or non-flat

        for each non-flat and not non-linear segment:
            if contains linear slope:
                mark slope as impulse or collapse
                for each flanking:
                    detetect non-linear type
            else
                mark as non-linear

        for each non-linear segment:
            if contains detectable non-linear type:
                mark type
            else:
                mark undefined

    Args:
        dydt:
        dydt_signs:
        ddydt_signs:
        phases:
        filt:
        offset:
        thresholds:

    Returns:

    """

    # TODO: Non linear phases should also have minimum lengths
    # TODO: Bridge 1 gaps if same on both sides

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    _set_flat_segments(dydt_signs,
                       thresholds[Thresholds.LinearModelMinimumLength],
                       phases)

    yield None

    while (phases == CurvePhases.UndeterminedNonFlat.value).any():

        flanking = _set_nonflat_linear_segment(
            dydt, dydt_signs,
            thresholds[Thresholds.LinearModelExtension],
            thresholds[Thresholds.LinearModelMinimumLength],
            phases)

        yield None

        if flanking.any():

            first_on_left_flank = flanking.argmin()

            for filt in _get_candidate_segment(flanking):

                direction = PhaseEdge.Right if \
                    filt.argmin() == first_on_left_flank else \
                    PhaseEdge.Left

                _set_nonlinear_phase_type(
                    dydt, dydt_signs, ddydt_signs, filt,
                    PhaseEdge.Left if direction is PhaseEdge.Right else PhaseEdge.Right,
                    thresholds[Thresholds.UniformityThreshold],
                    thresholds[Thresholds.UniformityTestSize])

                # Only look for the first non-linear segment rest is up for grabs for
                # Next iteration of finding impulses or collapses
                flanking[filt] = False
                yield None

    # Try to classify remaining positions as non linear phases
    while filt in _get_candidate_segment(phases, test_value=CurvePhases.UndeterminedNonLinear.value):

        phase = _set_nonlinear_phase_type(
            dydt, dydt_signs, ddydt_signs, filt,
            PhaseEdge.Intelligent,
            thresholds[Thresholds.UniformityThreshold],
            thresholds[Thresholds.UniformityTestSize])

        # If currently considered segment had no phase then it is undetermined
        if phase is CurvePhases.Undetermined:

            phases[filt] = phase.value

        yield  None

    # If there's an offset assume phase carries to edge
    if offset:
        phases[:offset] = phases[offset]
        phases[-offset:] = phases[-offset - 1]
        yield None


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
            yield  labels == 1
        else:
            break


def _set_flat_segments(dydt_signs, minimum_segmentlength, phases):

    phases[...] = CurvePhases.UndeterminedNonFlat.value
    flats = _bridge_canditates(dydt_signs == 0)
    for length, left, right in izip(*_get_candidate_lengths_and_edges(flats)):
        if length >= minimum_segmentlength:
            phases[left: right] = CurvePhases.Flat.value


def _get_candidate_lengths_and_edges(candidates):

    kernel = [-1, 1]
    edges = signal.convolve(candidates, kernel, mode='same')
    lefts, = np.where(edges == -1)
    rights, = np.where(edges == 1)
    if rights.size < lefts.size:
        rights = np.hstack((rights, candidates.size))

    return rights - lefts, lefts, rights


def _bridge_canditates(candidates, window_size=5):
    for window in range(3, window_size, 2):
        candidates = signal.medfilt(candidates, window_size).astype(bool) | candidates
    return candidates


def _set_nonflat_linear_segment(dydt, dydt_signs, extension_threshold, minimum_length_threshold, phases):

    # All positions with sufficient slope
    filt = (dydt_signs != 0) & (phases == CurvePhases.UndeterminedNonFlat.value)

    # In case there are small regions left
    if not filt.any():

        phases[phases == CurvePhases.UndeterminedNonFlat.value] = CurvePhases.UndeterminedNonLinear.value
        # Since no segment was detected there are no bordering segments
        return np.array([])

    # Determine value and position of steepest slope
    loc_value = np.abs(dydt[filt]).max()
    loc = np.where((np.abs(dydt) == loc_value) & filt)[0][0]
    loc_value = dydt[loc]

    # Determine comparison operator for first derivative
    phase = CurvePhases.Impulse if loc_value < 0 else CurvePhases.Collapse
    op1 = operator.lt if phase is CurvePhases.Collapse else operator.gt

    # Update filter to only keep slopes in the direction of the phase
    filt = op1(dydt_signs, 0) & (phases == CurvePhases.UndeterminedNonFlat.value)

    # Find all candidates
    candidates = op1(dydt, loc_value * extension_threshold) & filt
    candidates = _bridge_canditates(candidates)
    candidates, n_found = label(candidates)

    # Verify that there's actually still a candidate at the peak value
    if n_found == 0:

        # TODO: It should not be possible for this to happen so print just in case
        print("Error detecting linear non-flat segment with max {0}, segment disappeared".format(loc_value))

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

    # Set the detected phase
    if offset:
        phases[offset: -offset][elected] = phase.value
    else:
        phases[elected] = phase.value

    # Locate flanking segments
    border_candidates, _ = label(filt)
    loc_label = border_candidates[loc]
    return (labels == loc_label) - elected

def _locate_segment(filt):  # -> (int, int)
    """

    Args:
        filt: a boolean array

    Returns:
        Left and exclusive right indices of filter
    """
    labels, n = label(filt)
    if n == 1:
        where = np.where(labels == 1)[0]
        return where[0], where[-1] + 1
    elif n > 1:
        raise ValueError("Filter is not homogenous, contains {0} segments ({1})".format(n, labels.tolist()))
    else:
        return None, None


def _get_filter(left=None, right=None, filt=None, size=None):
    """

    Args:
        left: inclusive left index or None (sets index as 0)
        right: exclusive right index or None (sets at size of filt)
        filt: previous filter array to reuse
        size: if no previous filter is supplied, size determines filter creation

    Returns: Filter array
        :rtype: numpy.ndarray
    """
    if filt is None:
        filt = np.zeros(size).astype(bool)
    else:
        filt[:] = False
        size = filt.size

    if left is None:
        left = 0
    if right is None:
        right = size

    filt[left: right] = True
    return filt


def _custom_filt(v, max_gap=3, min_length=3):

    w, = np.where(v)
    if not w.any():
        return False
    filted = signal.convolve(v[w[0]:w[-1] + 1] == np.False_, (1,)*max_gap, mode='same') < max_gap
    padded = np.hstack([(0,), filted, (0,)]).astype(int)
    diff = np.diff(padded)
    return (np.where(diff < 0)[0] - np.where(diff > 0)[0]).max() >= min_length


def _set_nonlinear_phase_type(dydt, dydt_signs, ddydt_signs, filt, test_edge,
                              uniformity_threshold, test_length, phases):
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
        test_length: How many points should be tested as a maximum
        phases: The phase-classification arrya

    Returns: The phase type, any of the following
        CurvePhases.Undetermined,
        CurvePhases.GrowthAcceleration,
        CurvePhases.CollapseAcceleration,
        CurvePhases.GrowthRetardation,
        CurvePhases.CollapseRetardation

    """

    # Define type at one of the edges
    if test_edge is PhaseEdge.Intelligent:

        # This takes a rough estimate of which side is more interesting
        # based on the location of the steepest slope

        steepest_loc = np.abs(dydt[filt]).argmax()
        test_edge = PhaseEdge.Left if steepest_loc / float(filt.sum()) < 0.5 else PhaseEdge.Right

    if test_edge is PhaseEdge.Left:
        ddydt_section = ddydt_signs[filt][:test_length]
        dydt_section = dydt_signs[filt][:test_length]
    elif test_edge is PhaseEdge.Right:
        ddydt_section = ddydt_signs[filt][-test_length:]
        dydt_section = dydt_signs[filt][-test_length:]
    else:
        return CurvePhases.Undetermined

    if ddydt_section.size == 0:
        return CurvePhases.Undetermined

    # Classify as acceleration or retardation
    sign = ddydt_section.mean()
    if sign > uniformity_threshold:
        candidate_phase_types = (CurvePhases.GrowthAcceleration, CurvePhases.CollapseRetardation)
    elif sign < -uniformity_threshold:
        candidate_phase_types = (CurvePhases.GrowthRetardation, CurvePhases.CollapseAcceleration)
    else:
        return CurvePhases.Undetermined

    # Classify as growth or collapse
    sign = dydt_section.mean()
    if sign > uniformity_threshold:
        phase = candidate_phase_types[0]
    elif sign < -uniformity_threshold:
        phase = candidate_phase_types[1]
    else:
        return CurvePhases.Undetermined

    # Determine which operators to be used for first (op1) and second (op2) derivative signs
    if phase is CurvePhases.GrowthAcceleration or phase is CurvePhases.GrowthRetardation:
        op1 = operator.gt
    else:
        op1 = operator.lt

    if phase is CurvePhases.GrowthAcceleration or phase is CurvePhases.CollapseRetardation:
        op2 = operator.ge
    else:
        op2 = operator.le

    candidates = filt & op1(dydt_signs, 0) & op2(ddydt_signs, 0)
    candidates = generic_filter(candidates, _custom_filt, size=9, mode='nearest')
    candidates, label_count = label(candidates)

    if label_count:

        candidates = candidates == (1 if test_edge is PhaseEdge.Left else label_count)

        if offset:
            phases[offset: -offset][candidates] = phase.value
        else:
            phases[candidates] = phase.value

        return phase

    else:

        return CurvePhases.Undetermined


def _phenotype_phases(curve, derivative, phases, times, doublings):

    derivative_offset = (times.shape[0] - derivative.shape[0]) / 2
    phenotypes = []

    # noinspection PyTypeChecker
    for phase in CurvePhases:

        labels, label_count = label(phases == phase.value)
        for id_label in range(1, label_count + 1):

            if phase == CurvePhases.Undetermined or phase == CurvePhases.Multiple:
                phenotypes.append((phase, None))
                continue

            filt = labels == id_label
            left, right = _locate_segment(filt)
            time_right = times[right - 1]
            time_left = times[left]
            current_phase_phenotypes = {}

            if phase == CurvePhases.GrowthAcceleration or phase == CurvePhases.GrowthRetardation:
                # A. For non-linear phases use the X^2 coefficient as curvature measure

                # TODO: Resloved worst problem, might still be lurking, angles are surprisingly close to PI

                k1 = derivative[max(0, left - derivative_offset)]
                k2 = derivative[right - 1 - derivative_offset]
                m1 = np.log2(curve[left]) - k1 * time_left
                m2 = np.log2(curve[right - 1]) - k2 * time_right
                i_x = (m2 - m1) / (k1 - k2)
                current_phase_phenotypes[CurvePhasePhenotypes.AsymptoteIntersection] = \
                    (i_x - time_left) / (time_right - time_left)
                current_phase_phenotypes[CurvePhasePhenotypes.AsymptoteAngle] = \
                    np.pi + np.arctan2(k1, 1) - np.arctan2(k2, 1)

            else:
                # B. For linear phases get the doubling time
                slope, intercept, _, _, _ = linregress(times[filt], np.log2(curve[filt]))
                current_phase_phenotypes[CurvePhasePhenotypes.PopulationDoublingTime] = 1 / slope
                current_phase_phenotypes[CurvePhasePhenotypes.LinearModelSlope] = slope
                current_phase_phenotypes[CurvePhasePhenotypes.LinearModelIntercept] = intercept

            # C. Get duration
            current_phase_phenotypes[CurvePhasePhenotypes.Duration] = time_right - time_left

            # D. Get fraction of doublings
            current_phase_phenotypes[CurvePhasePhenotypes.FractionYield] = \
                (np.log2(curve[right - 1]) - np.log2(curve[left])) / doublings

            # E. Get start of phase
            current_phase_phenotypes[CurvePhasePhenotypes.Start] = time_left

            phenotypes.append((phase, current_phase_phenotypes))

    # Phenotypes sorted on phase start rather than type of phase
    return sorted(phenotypes, key=lambda (t, p): p[CurvePhasePhenotypes.Start] if p is not None else 9999)


def _get_data_needed_for_segments(phenotyper_object, plate, pos, threshold_for_sign, threshold_flatline):

    curve = phenotyper_object.smooth_growth_data[plate][pos]

    # Some center weighted smoothing of derivative, we only care for general shape
    dydt = signal.convolve(phenotyper_object.get_derivative(plate, pos), [0.1, 0.25, 0.3, 0.25, 0.1], mode='valid')
    d_offset = (phenotyper_object.times.size - dydt.size) / 2
    dydt = np.hstack(([dydt[0] for _ in range(d_offset)], dydt, [dydt[-1] for _ in range(d_offset)]))

    dydt_ranks = np.abs(dydt).argsort().argsort()
    offset = (phenotyper_object.times.shape[0] - dydt.shape[0]) / 2

    # Smoothing in kernel shape because only want reliable trends
    ddydt = signal.convolve(dydt, [1, 1, 1, 0, -1, -1, -1], mode='valid')
    dd_offset = (dydt.size - ddydt.size) / 2
    ddydt = np.hstack(([ddydt[0] for _ in range(dd_offset)], ddydt, [ddydt[-1] for _ in range(dd_offset)]))
    phases = np.ones_like(curve).astype(np.int) * 0
    """:type : numpy.ndarray"""
    filt = _get_filter(size=dydt.size)

    # Determine second derviative signs
    ddydt_signs = np.sign(ddydt)
    ddydt_signs[np.abs(ddydt) < threshold_for_sign * ddydt[np.isfinite(ddydt)].std()] = 0

    # Determine first derivative signs
    dydt_signs = np.sign(dydt)
    dydt_signs[np.abs(dydt) < threshold_flatline] = 0

    return dydt, dydt_ranks, dydt_signs, ddydt_signs, phases, filt, offset, curve


def phase_phenotypes(phenotyper_object, plate, pos, thresholds=None, experiment_doublings=None):

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    dydt, dydt_ranks,  dydt_signs, ddydt_signs, phases, filt, offset, curve = _get_data_needed_for_segments(
        phenotyper_object, plate, pos,
        thresholds[Thresholds.SecondDerivativeSigmaAsNotZero],
        thresholds[Thresholds.FlatlineSlopRequirement])

    for _ in _segment(dydt, dydt_signs, ddydt_signs, phases,
                      filt=filt, offset=offset, thresholds=thresholds):
        pass

    if experiment_doublings is None:
        experiment_doublings = (np.log2(phenotyper_object.get_phenotype(
            growth_phenotypes.Phenotypes.ExperimentEndAverage)[plate][pos]) -
                                np.log2(phenotyper_object.get_phenotype(
                                    growth_phenotypes.Phenotypes.ExperimentBaseLine)[plate][pos]))

    return phases, _phenotype_phases(curve, dydt, phases, phenotyper_object.times, experiment_doublings)


def filter_plate_custom_filter(
        plate,
        phase=CurvePhases.GrowthAcceleration,
        measure=CurvePhasePhenotypes.AsymptoteIntersection,
        phases_requirement=lambda phases: len(phases) == 1,
        phase_selector=lambda phases: phases[0]):

    def f(phenotype_vector):
        phases = tuple(d for t, d in phenotype_vector if t == phase)
        if phases_requirement(phases):
            return phase_selector(phases)[measure]
        return np.nan

    return np.ma.masked_invalid(np.frompyfunc(f, 1, 1)(plate).astype(np.float))


def filter_plate_on_phase_id(plate, phases_id, measure):

    def f(phenotype_vector, phase_id):
        if phase_id < 0:
            return np.nan

        try:
            return phenotype_vector[phase_id][1][measure]
        except (KeyError, TypeError):
            return np.nan

    return np.ma.masked_invalid(np.frompyfunc(f, 2, 1)(plate, phases_id).astype(np.float))


def _get_phase_id(plate, *phases):

    l = len(phases)

    def f(v):
        v = zip(*v)[0]
        i = 0
        for id_phase, phase in enumerate(v):
            if i < l:
                if phase is phases[i]:
                    i += 1
                    if i == l:
                        return id_phase

        return -1

    return np.frompyfunc(f, 1, 1)(plate).astype(np.int)


def _impulse_counter(phase_vector):
    if phase_vector:
        return sum(1 for phase in phase_vector if phase[0] == CurvePhases.Impulse)
    return -np.inf


def _collapse_counter(phase_vector):
    if phase_vector:
        return sum(1 for phase in phase_vector if phase[0] == CurvePhases.Collapse)
    return -np.inf


def filter_plate(plate, meta_phenotype, phenotypes):

    if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution or \
            meta_phenotype == CurvePhaseMetaPhenotypes.FirstMinorImpulseYieldContribution:

        index = -1 if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution else -2
        phase_need = 1 if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution else 2

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.FractionYield,
            phases_requirement=lambda phases: len(phases) >= phase_need,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.FractionYield] if
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[index]])

    elif (meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseAveragePopulationDoublingTime or
            meta_phenotype == CurvePhaseMetaPhenotypes.FirstMinorImpulseAveragePopulationDoublingTime):

        index = -1 if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseAveragePopulationDoublingTime else -2
        phase_need = 1 if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseAveragePopulationDoublingTime else 2

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.PopulationDoublingTime,
            phases_requirement=lambda phases: len(phases) >= phase_need,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.FractionYield] if
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[index]])

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialLag:

        flat_slope = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        flat_intercept = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        # TODO: Consider using major phase
        impules_phase = _get_phase_id(plate, CurvePhases.Flat, CurvePhases.Impulse)

        impulse_slope = filter_plate_on_phase_id(
            plate, impules_phase, measure=CurvePhasePhenotypes.LinearModelSlope)

        impulse_intercept = filter_plate_on_phase_id(
            plate, impules_phase, measure=CurvePhasePhenotypes.LinearModelIntercept)

        lag = (impulse_intercept - flat_intercept) / (flat_slope - impulse_slope)
        lag[lag < 0] = np.nan
        return np.ma.masked_invalid(lag)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialLagAlternativeModel:

        impulse_slope = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.FractionYield] if
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[-1]])

        impulse_intercept = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.FractionYield] if
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[-1]])

        impulse_start = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.Start,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.FractionYield] if
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[-1]])

        flat_slope = 0
        flat_intercept = phenotypes[..., growth_phenotypes.Phenotypes.ExperimentLowPoint.value]
        low_point_time = phenotypes[..., growth_phenotypes.Phenotypes.ExperimentLowPointWhen.value]

        lag = (impulse_intercept - np.log2(flat_intercept)) / (flat_slope - impulse_slope)

        lag[(lag < 0) | (impulse_start < low_point_time)] = np.nan

        return np.ma.masked_invalid(lag)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteAngle:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.GrowthAcceleration,
            measure=CurvePhasePhenotypes.AsymptoteAngle,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteAngle:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.GrowthRetardation,
            measure=CurvePhasePhenotypes.AsymptoteAngle,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[-1]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteIntersect:
        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.GrowthAcceleration,
            measure=CurvePhasePhenotypes.AsymptoteIntersection,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteIntersect:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.GrowthRetardation,
            measure=CurvePhasePhenotypes.AsymptoteIntersection,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[-1]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.Modalities:

        return np.ma.masked_invalid(np.frompyfunc(_impulse_counter, 1, 1)(plate).astype(np.float))

    elif meta_phenotype == CurvePhaseMetaPhenotypes.Collapses:

        return np.ma.masked_invalid(np.frompyfunc(_collapse_counter, 1, 1)(plate).astype(np.float))

    else:

        return np.ma.masked_invalid(np.ones_like(plate.shape) * np.nan)
