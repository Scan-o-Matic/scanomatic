import numpy as np
import operator
from scipy import signal
from scipy.ndimage import label, generic_filter
from scipy.stats import linregress
from collections import deque
from enum import Enum

from scanomatic.data_processing import growth_phenotypes

# TODO: Verify that all offsets work properly, especially when combining ddydt_sign an dydt


class CurvePhases(Enum):
    """Phases of curves recognized

    Attributes:
        CurvePhases.Multiple: Several types for same position, to be considered as an error.
        CurvePhases.Undetermined: Positions yet to be classified or not fulfilling any classification
        CurvePhases.Flat: Positions that exhibit no growth or collapse
        CurvePhases.Acceleration: Positions that are characterized by a positive second derivative.
        CurvePhases.Retardation: Positions that are characterized by a negative second derivative.
        CurvePhases.Impulse: Close to linear segment with growth.
        CurvePhases.Collapse: Close to linear segment with decreasing population size.
    """
    Multiple = -1
    """:type : CurvePhases"""
    Undetermined = 0
    """:type : CurvePhases"""
    Flat = 1
    """:type : CurvePhases"""
    Acceleration = 2
    """:type : CurvePhases"""
    Retardation = 3
    """:type : CurvePhases"""
    Impulse = 4
    """:type : CurvePhases"""
    Collapse = 5
    """:type : CurvePhases"""


class Thresholds(Enum):
    """Thresholds used by the phase algorithm

    Attributes:
        Thresholds.ImpulseExtension: Factor for impulse and collapse slopes to be considered equal to max/min point.
        Thresholds.ImpulseSlopeRequirement: Minimum slope to be impulse/collapse.
        Thresholds.FlatlineSlopRequirement: Maximum slope for something to be flatline.
        Thresholds.FractionAcceleration: The amount of second derivative signs to be of a certain type to classify as
            acceleration or deceleration.
        Thresholds.FractionAccelerationTestDuration: The number of measurements included in `FractionAcceleration` test.
    """
    ImpulseExtension = 0
    """:type : Thresholds"""
    ImpulseSlopeRequirement = 1
    """:type : Thresholds"""
    FlatlineSlopRequirement = 2
    """:type : Thresholds"""
    FractionAcceleration = 3
    """:type : Thresholds"""
    FractionAccelerationTestDuration = 4
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
    """
    Left = 0
    """:type : PhaseEdge"""
    Right = 1
    """:type : PhaseEdge"""


def _filter_find(vector, filt, func=np.max):
    vector = np.abs(vector)
    return np.where((vector == func(vector[filt])) & filt)[0]

DEFAULT_THRESHOLDS = {
    Thresholds.ImpulseExtension: 0.75,
    Thresholds.ImpulseSlopeRequirement: 0.02,
    Thresholds.FlatlineSlopRequirement: 0.02,
    Thresholds.FractionAcceleration: 0.66,
    Thresholds.FractionAccelerationTestDuration: 3}


def _verify_impulse_or_collapse(dydt, loc_max, thresholds, left, right, phases, offset):
    if np.abs(dydt[loc_max]) < thresholds[Thresholds.ImpulseSlopeRequirement]:
        if left == 0 and offset:
            phases[:offset] = phases[offset]

        if right == phases.size and offset:
            phases[-offset:] = phases[-offset - 1]
        return False
    return True


def _verify_impulse_or_collapse_though_growth_delta(impulse_left, impulse_right, left, right, phases, offset):

    if impulse_left is None or impulse_right is None:
        if left == 0 and offset:
            phases[:offset] = phases[offset]
        if right == phases.size and offset:
            phases[-offset:] = phases[-offset - 1]

        return True
    return False


def _test_phase_type(ddydt_signs, left, right, filt, test_edge, uniformity_threshold, selection_length):

    candidates = _get_filter(left, right, size=ddydt_signs, filt=filt)
    if test_edge is PhaseEdge.Left:
        selection = ddydt_signs[candidates][:selection_length]
    elif test_edge is PhaseEdge.Right:
        selection = ddydt_signs[candidates][-selection_length:]
    else:
        return CurvePhases.Undetermined

    if selection.size == 0:
        return CurvePhases.Undetermined

    sign = selection.mean()
    if sign > uniformity_threshold:
        return CurvePhases.Acceleration
    elif sign < -uniformity_threshold:
        return CurvePhases.Retardation
    else:
        return CurvePhases.Undetermined


def _verify_has_flat(dydt, filt, flat_threshold):

    candidates = (np.abs(dydt) < flat_threshold) & filt
    candidates = signal.medfilt(candidates, 3).astype(bool)
    return candidates.any()


def _get_linear_feature(test_funcs, test_argvs, eval_funcs, eval_argvs, left, right):

    if test_funcs:
        test_func = test_funcs.popleft()
        test_argv = test_argvs.popleft()
        eval_func = eval_funcs.popleft()
        eval_argv = eval_argvs.popleft()
        if test_func(*test_argv):
            try:
                return eval_func(*eval_argv)
            except ValueError:
                return _get_linear_feature(test_funcs, test_argvs, eval_funcs, eval_argvs, left, right)
        else:
            return _get_linear_feature(test_funcs, test_argvs, eval_funcs, eval_argvs, left, right)
    else:
        # NOTE: Should be inverted when no feature found
        return right, left


def _segment(dydt, dydt_ranks, ddydt_signs, phases, filt, offset, thresholds=None):

    if phases.all() or not filt.any():
        raise StopIteration

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if np.unique(phases[offset: -offset][filt] if offset else phases[filt]).size != 1:
        raise ValueError("Impossible to segment due to multiple phases {0} filter {1}".format(
            np.unique(phases[filt][offset: -offset] if offset else phases[filt]), filt
        ))

    # 1. Find segment's borders
    left, right = _locate_segment(filt)

    # 2. Find segment's maximum & min growth
    loc_max = _filter_find(dydt_ranks, filt)
    loc_min = _filter_find(dydt_ranks, filt, np.min)

    impulse_left, impulse_right = _get_linear_feature(
        deque((_verify_impulse_or_collapse, _verify_has_flat)),
        deque(((dydt, loc_max, thresholds, left, right, phases, offset),
               (dydt, filt, thresholds[Thresholds.FlatlineSlopRequirement]))),
        deque((_locate_impulse_or_collapse, _locate_flat)),
        deque(((dydt, loc_max, phases, filt, offset, thresholds[Thresholds.ImpulseExtension]),
               (dydt, loc_min, phases, filt, offset, thresholds[Thresholds.FlatlineSlopRequirement]))),
        left,
        right
    )

    yield None

    for direction, (l, r) in zip(PhaseEdge, ((left, impulse_left), (impulse_right, right))):

        if phases[l: r].all():
            continue

        phase = _test_phase_type(
            ddydt_signs, l, r, filt, PhaseEdge.Left if direction is PhaseEdge.Right else PhaseEdge.Right,
            thresholds[Thresholds.FractionAcceleration], thresholds[Thresholds.FractionAccelerationTestDuration])
        # print("Investigate {0} -> {1}".format(direction, phase))
        if phase is CurvePhases.Acceleration:
            # 5. Locate acceleration phase
            (phase_left, phase_right), _ = _locate_acceleration(
                dydt, ddydt_signs, phases, l, r, offset,
                flatline_threshold=thresholds[Thresholds.FlatlineSlopRequirement])

            yield None

        elif phase is CurvePhases.Retardation:

            # 6. Locate retardation phase
            (phase_left, phase_right), _ = _locate_retardation(
                dydt, ddydt_signs, phases, l, r, offset,
                flatline_threshold=thresholds[Thresholds.FlatlineSlopRequirement])

            yield None

        else:
            # No phase found
            phase_left = r
            phase_right = l

        # 7. If there's anything remaining on left, investigated for more impulses/collapses
        if direction is PhaseEdge.Left and right != phase_left:
            # print "Left investigate"
            for ret in _segment(dydt, dydt_ranks, ddydt_signs, phases,
                                _get_filter(left, phase_left, size=dydt.size, filt=filt), offset, thresholds):

                yield ret

            yield None

        # 8. If there's anything remaining right, investigate for more impulses/collapses
        if direction is PhaseEdge.Right and left != phase_right:
            # print "Right investigate"
            for ret in _segment(dydt, dydt_ranks, ddydt_signs, phases,
                                _get_filter(phase_right, right, size=dydt.size, filt=filt), offset, thresholds):
                yield ret

            yield None

    # 9. Update phases edges
    if offset:
        phases[:offset] = phases[offset]
        phases[-offset:] = phases[-offset - 1]
        yield None


def _locate_flat(dydt, loc, phases, filt, offset, extension_threshold):

    candidates = (np.abs(dydt) < extension_threshold) & filt
    candidates = signal.medfilt(candidates, 3).astype(bool)
    candidates, n_found = label(candidates)
    if candidates[loc] == 0:
        if n_found == 0:
            raise ValueError("Least slope {0}, loc {1} is not a candidate {2} (filt {3})".format(
                dydt[loc], loc, candidates.tolist(), filt.tolist()))
        else:
            loc = np.where((dydt[candidates > 0].argmin() == dydt) & (candidates > 0))[0]
    if offset:
        phases[offset: -offset][candidates == candidates[loc]] = CurvePhases.Flat.value
    else:
        phases[candidates == candidates[loc]] = CurvePhases.Flat.value

    return _locate_segment(candidates == candidates[loc])


def _locate_impulse_or_collapse(dydt, loc, phases, filt, offset, extension_threshold):

    phase = CurvePhases.Impulse if np.sign(dydt[loc]) > 0 else CurvePhases.Collapse
    comp = operator.gt if phase is CurvePhases.Impulse else operator.lt

    candidates = comp(dydt, dydt[loc] * extension_threshold) & filt
    candidates = signal.medfilt(candidates, 3).astype(bool)

    candidates, _ = label(candidates)
    if offset:
        phases[offset: -offset][candidates == candidates[loc]] = phase.value
    else:
        phases[candidates == candidates[loc]] = phase.value

    return _locate_segment(candidates == candidates[loc])


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


def _locate_acceleration(dydt, ddydt_signs, phases, left, right, offset, flatline_threshold, filt=None):

    candidates = _get_filter(left, right, size=dydt.size, filt=filt)
    candidates2 = candidates & (np.abs(dydt) > flatline_threshold) & (ddydt_signs == 1)

    candidates2 = generic_filter(candidates2, _custom_filt, size=9, mode='nearest')
    candidates2, label_count = label(candidates2)

    if label_count:
        acc_candidates = candidates2 == label_count
        if offset:
            phases[offset: -offset][acc_candidates] = CurvePhases.Acceleration.value
        else:
            phases[acc_candidates] = CurvePhases.Acceleration.value

        return _locate_segment(acc_candidates), CurvePhases.Flat
    else:
        if offset:
            phases[offset: -offset][candidates] = CurvePhases.Undetermined.value
        else:
            phases[candidates] = CurvePhases.Undetermined.value

        return (left, right), CurvePhases.Undetermined


def _locate_retardation(dydt, ddydt_signs, phases, left, right, offset, flatline_threshold, filt=None):

    candidates = _get_filter(left, right, size=dydt.size, filt=filt)
    candidates2 = candidates & (np.abs(dydt) > flatline_threshold) & (ddydt_signs == -1)

    candidates2 = generic_filter(candidates2, _custom_filt, size=9, mode='nearest')
    candidates2, label_count = label(candidates2)

    if label_count:
        ret_cantidates = candidates2 == 1
        if offset:
            phases[offset: -offset][ret_cantidates] = CurvePhases.Retardation.value
        else:
            phases[ret_cantidates] = CurvePhases.Retardation.value

        return _locate_segment(ret_cantidates), CurvePhases.Flat
    else:
        if offset:
            phases[offset: -offset][candidates] = CurvePhases.Undetermined.value
        else:
            phases[candidates] = CurvePhases.Undetermined.value
        return (left, right), CurvePhases.Undetermined


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

            if phase == CurvePhases.Acceleration or phase == CurvePhases.Retardation:
                # A. For non-linear phases use the X^2 coefficient as curvature measure

                a1 = np.array((time_left, np.log2(curve[left])))
                a2 = np.array((time_right, np.log2(curve[right - 1])))
                k1 = derivative[left - derivative_offset]
                k2 = derivative[right - 1 - derivative_offset]
                m1 = a1[1] - k1 * a1[0]
                m2 = a2[1] - k2 * a2[1]
                i_x = (m2 - m1) / (k1 - k2)
                i = np.array((i_x, k1 * i_x + m1))
                a1 -= i
                a2 -= i
                current_phase_phenotypes[CurvePhasePhenotypes.AsymptoteIntersection] = \
                    (i_x - time_left) / (time_right - time_left)
                current_phase_phenotypes[CurvePhasePhenotypes.AsymptoteAngle] = \
                    np.arctan((k2 - k1) / (1 + k1 * k2))

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


def _get_data_needed_for_segments(phenotyper_object, plate, pos):

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
    return dydt, dydt_ranks, np.sign(ddydt), phases, filt, offset, curve


def phase_phenotypes(phenotyper_object, plate, pos, thresholds=None, experiment_doublings=None):

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    dydt, dydt_ranks, ddydt_signs, phases, filt, offset, curve = _get_data_needed_for_segments(
        phenotyper_object, plate, pos)

    for _ in _segment(dydt, dydt_ranks, ddydt_signs, phases, filt=filt, offset=offset, thresholds=thresholds):
        pass

    if experiment_doublings is None:
        experiment_doublings = (np.log2(phenotyper_object.get_phenotype(
            growth_phenotypes.Phenotypes.ExperimentEndAverage)[plate][pos]) -
                                np.log2(phenotyper_object.get_phenotype(
                                    growth_phenotypes.Phenotypes.ExperimentBaseLine)[plate][pos]))

    return phases, _phenotype_phases(curve, dydt, phases, phenotyper_object.times, experiment_doublings)


def filter_plate_custom_filter(
        plate,
        phase=CurvePhases.Acceleration,
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
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[0]])

        impulse_intercept = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.FractionYield] if
                phase[CurvePhasePhenotypes.FractionYield] else -np.inf for phase in phases))[0]])

        flat_slope = 0
        flat_intercept = phenotypes[..., growth_phenotypes.Phenotypes.ExperimentLowPoint.value]

        lag = (impulse_intercept - np.log2(flat_intercept)) / (flat_slope - impulse_slope)

        # TODO: Ensure flat-measure occurs before impulse

        lag[lag < 0] = np.nan
        return np.ma.masked_invalid(lag)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteAngle:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Acceleration,
            measure=CurvePhasePhenotypes.AsymptoteAngle,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteAngle:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Retardation,
            measure=CurvePhasePhenotypes.AsymptoteAngle,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[-1]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialAccelerationAsymptoteIntersect:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Acceleration,
            measure=CurvePhasePhenotypes.AsymptoteIntersection,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.FinalRetardationAsymptoteIntersect:

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Retardation,
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
