import numpy as np
from enum import Enum
from itertools import izip

from scanomatic.data_processing import growth_phenotypes
from scanomatic.io.logger import Logger
from scanomatic.data_processing.phases.analysis import CurvePhasePhenotypes, number_of_phenotypes, get_phenotypes_tuple
from scanomatic.data_processing.phases.segmentation import CurvePhases, is_detected_non_linear

_l = Logger("Curve Phase Meta Phenotyping")


class CurvePhaseMetaPhenotypes(Enum):
    """Phenotypes of an entire growth-log2_curve based on the phase segmentation.

    Attributes:
        CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution:
            The fraction of the total yield (in population doublings) that the
            `CurvePhases.Impulse` that contribute most to the total yield is
            responsible for (`CurvePhasePhenotypes.PopulationDoublings`).

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

        CurvePhaseMetaPhenotypes.MajorImpulseFlankAsymmetry:
            The `CurvePhasePhenotypes.AsymptoteAngle` ratio of the right
            to left flanking non-linear phase.

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
        CurvePhaseMetaPhenotypes.Modalities:
            The number of `CurvePhases.Impulse`
        CurvePhaseMetaPhenotypes.Collapses:
            The number of `CurvePhases.Collapse`

    See Also:
        filter_plate: Get one of these out of a plate of phase segmentation information
    """
    MajorImpulseYieldContribution = 0
    """:type : CurvePhaseMetaPhenotypes """
    FirstMinorImpulseYieldContribution = 1
    """:type : CurvePhaseMetaPhenotypes """
    MajorImpulseAveragePopulationDoublingTime = 5
    """:type : CurvePhaseMetaPhenotypes """
    FirstMinorImpulseAveragePopulationDoublingTime = 6
    """:type : CurvePhaseMetaPhenotypes """
    MajorImpulseFlankAsymmetry = 8
    """:type : CurvePhaseMetaPhenotypes """

    InitialAccelerationAsymptoteAngle = 10
    """:type : CurvePhaseMetaPhenotypes """
    FinalRetardationAsymptoteAngle = 11
    """:type : CurvePhaseMetaPhenotypes """
    InitialAccelerationAsymptoteIntersect = 15
    """:type : CurvePhaseMetaPhenotypes """
    FinalRetardationAsymptoteIntersect = 16
    """:type : CurvePhaseMetaPhenotypes """

    InitialLag = 20
    """:type : CurvePhaseMetaPhenotypes """
    InitialLagAlternativeModel = 22
    """:type : CurvePhaseMetaPhenotypes """
    TimeBeforeMajorGrowth = 23
    """:type : CurvePhaseMetaPhenotypes """

    Modalities = 25
    """:type : CurvePhaseMetaPhenotypes """
    ModalitiesAlternativeModel = 27
    """:type : CurvePhaseMetaPhenotypes """

    Collapses = 26
    """:type : CurvePhaseMetaPhenotypes """


class VectorPhenotypes(Enum):
    """The vector type phenotypes used to store phase segmentation

    Attributes:
        VectorPhenotypes.PhasesClassifications:
            1D vector the same length as growth data with the `CurvePhases` values
            for classification of which phase each population size measurement in the growth data
            is classified as.
        VectorPhenotypes.PhasesPhenotypes:
            1D vector of `CurvePhasePhenotypes` keyed dicts for each segment in the log2_curve.
    """
    PhasesClassifications = 0
    """:type : VectorPhenotypes"""
    PhasesPhenotypes = 1
    """:type : VectorPhenotypes"""


def filter_plate_custom_filter(
        plate,
        phase=CurvePhases.GrowthAcceleration,
        measure=CurvePhasePhenotypes.AsymptoteIntersection,
        phases_requirement=lambda phases: len(phases) == 1,
        phase_selector=lambda phases: phases[0]):

    def f(phenotype_vector):

        try:
            phases = tuple(d for t, d in phenotype_vector if t == phase)
            if phases_requirement(phases):
                return phase_selector(phases)[measure]
        except TypeError:
            pass
        return np.nan

    return np.frompyfunc(f, 1, 1)(plate).astype(float)


def filter_plate_on_phase_id(plate, phases_id, measure):

    def f(phenotype_vector, phase_id):
        if phase_id < 0:
            return np.nan

        try:
            return phenotype_vector[phase_id][1][measure]
        except (KeyError, TypeError):
            return np.nan

    return np.frompyfunc(f, 2, 1)(plate, phases_id).astype(np.float)


def _get_phase_id(plate, *phases):

    l = len(phases)

    def f(v):
        try:
            v = zip(*v)[0]
            i = 0
            for id_phase, phase in enumerate(v):
                if i < l:
                    if phase is phases[i]:
                        i += 1
                        if i == l:
                            return id_phase
        except TypeError:
            pass
        return -1

    return np.frompyfunc(f, 1, 1)(plate).astype(np.int)


def _phase_finder(phase_vector, phase):

    try:
        return tuple(i for i, (p_type, p_data) in enumerate(phase_vector) if p_type == phase)
    except TypeError:
        return tuple()

# REGION: Phase counters


def _py_impulse_counter(phase_vector):
    try:
        return sum(1 for phase in phase_vector if phase[0] == CurvePhases.Impulse)
    except TypeError:
        return -1

_np_impulse_counter = np.frompyfunc(_py_impulse_counter, 1, 1)


def _np_ma_impulse_counter(phases):

    data = _np_impulse_counter(phases)
    data[data < 0] = np.nan
    return data


def _py_inner_impulse_counter(phase_vector):

    try:
        acc = _phase_finder(phase_vector, CurvePhases.GrowthAcceleration)
        if not acc:
            return -1
        ret = _phase_finder(phase_vector, CurvePhases.GrowthRetardation)
        if not ret:
            return -1
        return _py_impulse_counter(phase_vector[acc[0]: ret[-1]])
    except TypeError:
        return -1

_np_inner_impulse_counter = np.frompyfunc(_py_inner_impulse_counter, 1, 1)


def _np_ma_inner_impulse_counter(phases):

    data = _np_inner_impulse_counter(phases).astype(float)
    data[data < 0] = np.nan
    return data


def _py_collapse_counter(phase_vector):
    try:
        return sum(1 for phase in phase_vector if phase[0] == CurvePhases.Collapse)
    except TypeError:
        return -1

_np_collapse_counter = np.frompyfunc(_py_collapse_counter, 1, 1)


def _np_ma_collapse_counter(phases):

    data = _np_collapse_counter(phases)
    data[data < 0] = np.nan
    return data


def _py_phase_counter(phase_vector):

    return sum(1 for t, d in phase_vector if t is not CurvePhases.Undetermined)

_np_phase_counter = np.frompyfunc(_py_phase_counter, 1, 1)

# END REGION: Phase counters

# REGION: Major pulse index


def _py_get_major_impulse_for_plate(phases):
    """Locates major impulses

    First the phases sort order based on yield is constructed

    The indices and sort order of those that are impulses are
    collected.

    Then the original index of the phase with the highest
    sort order is returned.

    Args:
        phases: Plate of phase data

    Returns: 2D numpy.ndarray with indices of the major
        growth impulses in the vectors.
    """

    try:

        sort_order = np.argsort(tuple(
            p_data[CurvePhasePhenotypes.PopulationDoublings] if
            p_data is not None and p_data[CurvePhasePhenotypes.PopulationDoublings] else -np.inf
            for p_type, p_data in phases
        ))

        impulses = np.array(tuple(
            (i, v) for i, v in enumerate(sort_order) if
            phases[i][VectorPhenotypes.PhasesClassifications.value] == CurvePhases.Impulse))

        if impulses.any():
            return impulses[np.argmax(impulses[:, -1])][0]
    except TypeError:
        pass
    return -1

_np_get_major_impulse_for_plate = np.frompyfunc(_py_get_major_impulse_for_plate, 1, 1)


def _np_ma_get_major_impulse_indices(phases):

    data = _np_get_major_impulse_for_plate(phases)
    data[data < 0] = np.nan
    return data

# END REGION: Major pulse index


def _py_get_flanking_angle_relation(phases, major_impulse_index):

    def _flank_angle(flank, impulse):

        if flank is None:

            return np.arctan2(1,
                              impulse[VectorPhenotypes.PhasesPhenotypes.value][CurvePhasePhenotypes.LinearModelSlope])

        elif flank[VectorPhenotypes.PhasesClassifications.value] is CurvePhases.Flat:

            return np.pi - np.abs(
                np.arctan2(1, impulse[VectorPhenotypes.PhasesPhenotypes.value][CurvePhasePhenotypes.LinearModelSlope]) -
                np.arctan2(1, flank[VectorPhenotypes.PhasesPhenotypes.value][CurvePhasePhenotypes.LinearModelSlope]))

        elif is_detected_non_linear(flank[VectorPhenotypes.PhasesClassifications.value]):

            return flank[VectorPhenotypes.PhasesPhenotypes.value][CurvePhasePhenotypes.AsymptoteAngle]

        else:
            return np.inf

    if np.isnan(major_impulse_index) or \
            phases[major_impulse_index][VectorPhenotypes.PhasesPhenotypes.value] is None:
        return np.inf
    if phases[major_impulse_index][VectorPhenotypes.PhasesClassifications.value] is not CurvePhases.Impulse:
        _l.error("Got index {0} as Impulse but is {1} in {2}".format(
            major_impulse_index,
            phases[major_impulse_index][VectorPhenotypes.PhasesClassifications.value],
            phases))
        return np.inf

    a1 = _flank_angle(phases[major_impulse_index - 1] if major_impulse_index > 0 else None,
                      phases[major_impulse_index])

    a2 = _flank_angle(phases[major_impulse_index + 1] if major_impulse_index < len(phases) - 1 else None,
                      phases[major_impulse_index])

    return a2 / a1

_np_get_flanking_angle_relation = np.frompyfunc(_py_get_flanking_angle_relation, 2, 1)


def extract_phenotypes(plate, meta_phenotype, phenotypes):

    if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution or \
            meta_phenotype == CurvePhaseMetaPhenotypes.FirstMinorImpulseYieldContribution:

        index = -1 if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution else -2
        phase_need = 1 if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution else 2

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.PopulationDoublings,
            phases_requirement=lambda phases: len(phases) >= phase_need,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.PopulationDoublings] if
                phase[CurvePhasePhenotypes.PopulationDoublings] else -np.inf for phase in phases))[index]])

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
                phase[CurvePhasePhenotypes.PopulationDoublings] if
                phase[CurvePhasePhenotypes.PopulationDoublings] else -np.inf for phase in phases))[index]])

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialLag:

        flat_slope = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        flat_intercept = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        impulses_phase = _get_phase_id(plate, CurvePhases.Flat, CurvePhases.Impulse)

        impulse_slope = filter_plate_on_phase_id(
            plate, impulses_phase, measure=CurvePhasePhenotypes.LinearModelSlope)

        impulse_intercept = filter_plate_on_phase_id(
            plate, impulses_phase, measure=CurvePhasePhenotypes.LinearModelIntercept)

        lag = (impulse_intercept - flat_intercept) / (flat_slope - impulse_slope)
        lag[lag < 0] = np.nan
        return lag

    elif meta_phenotype == CurvePhaseMetaPhenotypes.TimeBeforeMajorGrowth:

        flat_slope = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        flat_intercept = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        impulses_phase = _np_ma_get_major_impulse_indices(plate)

        impulse_slope = filter_plate_on_phase_id(
            plate, impulses_phase, measure=CurvePhasePhenotypes.LinearModelSlope)

        impulse_intercept = filter_plate_on_phase_id(
            plate, impulses_phase, measure=CurvePhasePhenotypes.LinearModelIntercept)

        lag = (impulse_intercept - flat_intercept) / (flat_slope - impulse_slope)
        lag[lag < 0] = np.nan
        return lag

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialLagAlternativeModel:

        impulse_slope = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.PopulationDoublings] if
                phase[CurvePhasePhenotypes.PopulationDoublings] else -np.inf for phase in phases))[-1]])

        impulse_intercept = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.PopulationDoublings] if
                phase[CurvePhasePhenotypes.PopulationDoublings] else -np.inf for phase in phases))[-1]])

        impulse_start = filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.Start,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases:
            phases[np.argsort(tuple(
                phase[CurvePhasePhenotypes.PopulationDoublings] if
                phase[CurvePhasePhenotypes.PopulationDoublings] else -np.inf for phase in phases))[-1]])

        flat_slope = 0
        flat_intercept = phenotypes[growth_phenotypes.Phenotypes.ExperimentLowPoint]
        low_point_time = phenotypes[growth_phenotypes.Phenotypes.ExperimentLowPointWhen]

        lag = (impulse_intercept - np.log2(flat_intercept)) / (flat_slope - impulse_slope)

        lag[(lag < 0) | (impulse_start < low_point_time) | (~np.isfinite(low_point_time))] = np.nan

        return lag

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

        return _np_ma_impulse_counter(plate)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.ModalitiesAlternativeModel:

        return _np_ma_inner_impulse_counter(plate)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.Collapses:

        return _np_ma_collapse_counter(plate)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseFlankAsymmetry:

        indices = _np_ma_get_major_impulse_indices(plate)
        return _np_get_flanking_angle_relation(plate, indices).astype(np.float)

    else:
        _l.error("Not implemented phenotype extraction: {0}".format(meta_phenotype))
        return np.ones_like(plate) * np.nan


def get_phase_assignment_data(phenotypes, plate):

    data = []
    vshape = None
    for x, y in phenotypes.enumerate_plate_positions(plate):
        v = phenotypes.get_curve_phases(plate, x, y)
        if v is not None and v.ndim == 1 and v.shape[0] and (vshape is None or v.shape == vshape):
            if vshape is None:
                vshape = v.shape
            data.append(v)
    return np.ma.array(data)


def get_phase_assignment_frequencies(phenotypes, plate):

    data = get_phase_assignment_data(phenotypes, plate)
    min_length = data.max() + 1
    bin_counts = [np.bincount(data[..., i], minlength=min_length) for i in range(data.shape[1])]
    return np.array(bin_counts)


def get_variance_decomposition_by_phase(plate_phenotype, phenotypes, id_plate, id_time, min_members=0):

    filt = phenotypes.get_curve_qc_filter(id_plate)
    plate = np.ma.masked_array(plate_phenotype, filt)
    ret = {None: plate.ravel().var()}
    phases = phenotypes.get_curve_phases_at_time(id_plate, id_time)
    ret.update({phase: plate[phases == phase.value].ravel().var() for phase in CurvePhases if
                (phases == phase.value).sum() > min_members})
    return ret


def _get_index_array(shape):

    m = np.mgrid[:shape[0], :shape[1]]

    l = zip(*(v.ravel() for v in m))
    a2 = np.empty(m.shape[1:], dtype=np.object)
    a2.ravel()[:] = l
    return a2


class PhaseData(Enum):
    Type = 0
    """:type: PhaseData"""
    Members = 1
    """:type: PhaseData"""
    Anchor = 2
    """:type: PhaseData"""


class PhaseSide(Enum):
    Both = 0
    """:type: PhaseSide"""
    Left = 1
    """:type: PhaseSide"""
    Right = 2
    """:type: PhaseSide"""


def get_phase_phenotypes_aligned(phenotypes, plate):

    # TODO: 1. Make own module
    # TODO: 2. Support multiple plates and files, for this the global end_time should be used

    phases = []

    def current_phase(phase_ref):

        for i, phase in enumerate(phases):
            if phase_ref in phase[PhaseData.Members]:
                return i
        return None

    def insert_phase(phase_phenotypes, id_tup, prev_phase, side, end_time, major_phase_time):

        possible = get_possible(prev_phase, side)

        try:
            start = phase_phenotypes[1][CurvePhasePhenotypes.Start] / \
                major_phase_time - 1.0 if phase_phenotypes[1][CurvePhasePhenotypes.Start] < major_phase_time else \
                (phase_phenotypes[1][CurvePhasePhenotypes.Start] - major_phase_time) / (end_time - major_phase_time)

            anchor = start + (
                0.5 * phase_phenotypes[1][CurvePhasePhenotypes.Duration] / major_phase_time if start < 0 else
                0.5 * phase_phenotypes[1][CurvePhasePhenotypes.Duration] / (end_time - major_phase_time))
        except (TypeError, KeyError, IndexError):
            print(phase_phenotypes)
            raise

        phase_id = None
        for phase_id in possible:
            if phases[phase_id][PhaseData.Anchor] > anchor:
                break
        if phase_id is None:
            append_phases(phase_phenotypes, id_tup, end_time, major_phase_time)
        else:
            phases.insert(phase_id, {PhaseData.Type: phase_phenotypes[0], PhaseData.Members: set()})
            add_to_phase(phase_phenotypes, id_tup, phases[phase_id], end_time, major_phase_time)

    def get_possible(prev_phase, side):
        if side is PhaseSide.Both:
            return range(0 if prev_phase is None else prev_phase, len(phases))
        elif side is PhaseSide.Left:
            return range(0 if prev_phase is None else prev_phase, major_phase_id)
        else:
            return range(max((major_phase_id + 1, 0 if prev_phase is None else prev_phase), len(phases)))

    def optimal_phase(phase_phenotypes, phase_ref, prev_phase, side, end_time, major_phase_time):

        possible = get_possible(prev_phase, side)
        min_e = None
        best_id = None
        if phase_ref:
            if phase_ref[1] >= len(phases):
                return None

            min_e = get_energy(phases[phase_ref[1]], phase_phenotypes, end_time, major_phase_time)

            best_id = phase_ref[1]

        for phase_id in possible:

            # TODO: Somehow can fall outside `phases`
            energy = get_energy(phases[phase_id], phase_phenotypes, end_time, major_phase_time)
            if energy < 1 and (min_e is None or energy < min_e):
                min_e = energy
                best_id = phase_id

        return best_id

    def add_to_phase(phase_phenotypes, phase_ref, phase, end_time, major_phase_time, w=0.9):

        try:
            start = phase_phenotypes[1][CurvePhasePhenotypes.Start] / \
                major_phase_time - 1.0 if phase_phenotypes[1][CurvePhasePhenotypes.Start] < major_phase_time else \
                (phase_phenotypes[1][CurvePhasePhenotypes.Start] - major_phase_time) / (end_time - major_phase_time)

            anchor = start + (
                0.5 * phase_phenotypes[1][CurvePhasePhenotypes.Duration] / major_phase_time if start < 0 else
                0.5 * phase_phenotypes[1][CurvePhasePhenotypes.Duration] / (end_time - major_phase_time))
        except KeyError:
            print (phase_phenotypes)
            raise

        if PhaseData.Anchor in phase:
            phase[PhaseData.Anchor] = w * phase[PhaseData.Anchor] + (1 - w) * anchor
        else:
            phase[PhaseData.Anchor] = anchor

        phase[PhaseData.Members].add(phase_ref)

    def append_phases(data, phase_ref, end_time, major_phase_time):

        for phase in CurvePhases:

            if phase is CurvePhases.Undetermined or phase is not data[0]:
                continue

            phases.append({PhaseData.Type: phase, PhaseData.Members: set()})
            if data[0] is phase:
                add_to_phase(data, phase_ref, phases[-1], end_time, major_phase_time)

    def get_energy(phase, phase_phenotypes, end_time, major_phase_time):

        if phase[PhaseData.Type] is not phase_phenotypes[0]:
            return np.inf

        start = phase_phenotypes[1][CurvePhasePhenotypes.Start] / \
            major_phase_time - 1.0 if phase_phenotypes[1][CurvePhasePhenotypes.Start] < major_phase_time else \
            (phase_phenotypes[1][CurvePhasePhenotypes.Start] - major_phase_time) / (end_time - major_phase_time)

        end = start + (
            phase_phenotypes[1][CurvePhasePhenotypes.Duration] / major_phase_time if start < 0 else
            phase_phenotypes[1][CurvePhasePhenotypes.Duration] / (end_time - major_phase_time))

        phase_anchor = phase[PhaseData.Anchor] if PhaseData.Anchor in phase else None

        if phase_anchor is None:
            return 0
        elif start <= phase_anchor <= end:
            return 0
        else:
            return min((abs(v) for v in (phase_anchor - end, phase_anchor - start))) / float(end - start)

    end_time = phenotypes.times.max()
    plate_data = phenotypes._vector_phenotypes[plate][VectorPhenotypes.PhasesPhenotypes]
    filt = phenotypes.get_curve_qc_filter(plate)
    coords = _get_index_array(plate_data.shape)

    plate_data = plate_data[filt == np.False_]
    coords = coords[filt == np.False_]

    major_idx = np.ma.masked_invalid(_np_ma_get_major_impulse_indices(plate_data).astype(np.float))

    plate_data = plate_data[major_idx.mask == np.False_]
    coords = coords[major_idx.mask == np.False_]
    major_idx = major_idx[major_idx.mask == np.False_]

    l = _np_phase_counter(plate_data)
    id_most_left_phases = major_idx.argmax()
    id_most_right_phases = (l - major_idx).argmax()
    major_idx = [int(v) if np.isfinite(v) else None for v in major_idx]

    # Init left phases
    v = plate_data[id_most_left_phases]
    major_phase_time = v[major_idx[id_most_left_phases]][1][CurvePhasePhenotypes.Start] + \
        0.5 * v[major_idx[id_most_left_phases]][1][CurvePhasePhenotypes.Duration]
    for id_phase, phase_data in enumerate(plate_data[id_most_left_phases][: major_idx[id_most_left_phases] if
                                          isinstance(major_idx[id_most_left_phases], int) else None]):

        append_phases(phase_data, (id_most_left_phases, id_phase), end_time, major_phase_time)

    # Adding a major phase
    major_phase_id = len(phases)
    phases.append({PhaseData.Type: CurvePhases.Impulse, PhaseData.Members: set()})
    add_to_phase(plate_data[id_most_left_phases][major_idx[id_most_left_phases]],
                 (id_most_left_phases, major_idx[id_most_left_phases]),
                 phases[major_phase_id], end_time, major_phase_time)

    # Init right phases
    v = plate_data[id_most_right_phases]
    major_phase_time = v[major_idx[id_most_right_phases]][1][CurvePhasePhenotypes.Start] + \
        0.5 * v[major_idx[id_most_right_phases]][1][CurvePhasePhenotypes.Duration]

    for id_phase, phase_data in enumerate(v):
        if id_phase <= major_idx[id_most_right_phases]:
            continue
        append_phases(phase_data, (id_most_right_phases, id_phase), end_time, major_phase_time)

    # Run through all curves
    first_run = True

    for n in range(10):

        for id_curve, v in enumerate(plate_data):

            prev_phase = None
            major_phase = (id_curve, major_idx[id_curve])
            side = PhaseSide.Left if isinstance(major_phase[1], int) else PhaseSide.Both
            major_phase_time = None if side is PhaseSide.Both else \
                v[major_phase[1]][1][CurvePhasePhenotypes.Start] + \
                0.5 * v[major_phase[1]][1][CurvePhasePhenotypes.Duration]

            for id_phase, phase_data in enumerate(v):

                if phase_data is None or phase_data[1] is None:
                    continue

                id_tup = (id_curve, id_phase)
                cur_phase = current_phase(id_tup)

                if side is not PhaseSide.Both:
                    side = PhaseSide.Left if cur_phase < major_phase else PhaseSide.Right

                # May not move major phase alignment
                if id_tup == major_phase:
                    if first_run and cur_phase is None:
                        add_to_phase(phase_data, id_tup, phases[major_phase_id], end_time, major_phase_time)
                    prev_phase = cur_phase
                    continue

                if cur_phase is not None:
                    if cur_phase > prev_phase:
                        e = get_energy(phases[cur_phase], phase_data, end_time, major_phase_time)
                        if e == 0:
                            prev_phase = cur_phase
                            continue
                        phases[cur_phase][PhaseData.Members].remove(id_tup)

                best_phase = optimal_phase(phase_data, id_tup, prev_phase, side, end_time, major_phase_time)
                if best_phase is None:
                    insert_phase(phase_data, id_tup, prev_phase, side, end_time, major_phase_time)
                else:
                    add_to_phase(phase_data, id_tup, phases[best_phase], end_time, major_phase_time)

                prev_phase = cur_phase

        phases = [phase for phase in sorted(phases, key=lambda x: x[PhaseData.Anchor])
                  if len(phase[PhaseData.Members]) > (0 if n == 9 else int(0.05 * coords.size))]

        first_run = False
        # TODO: Should iterate until energy is stable

    return _ravel_phase_phenotypes(phases, plate_data, coords, phenotypes[plate].shape[:2])


def _ravel_phase_phenotypes(phases, ravel_plate, coords, shape):

    def ravel(data, coord, id_curve, phase_vector):

        for id_phase, (phase_type, phase_phenotypes) in enumerate(phase_vector):
            id_tup = (id_curve, id_phase)
            for id_data, aligned_phase in enumerate(phases):
                if id_tup in aligned_phase[PhaseData.Members]:
                    data[coord][idx[id_data]: idx[id_data] + number_of_phenotypes(phase_type)] = [
                        phase_phenotypes[key] for key in CurvePhasePhenotypes if key in phase_phenotypes
                    ]

    idx = [number_of_phenotypes(phase[PhaseData.Type]) for phase in phases]
    idx.insert(0, 0)
    idx = np.cumsum(idx)
    data = np.ones(shape + (idx[-1],), dtype=float) * np.nan

    for id_curve, (coord, phase_vector) in enumerate(izip(coords, ravel_plate)):
        ravel(data, coord, id_curve, phase_vector)

    return data, \
        tuple((phase[PhaseData.Type],
               get_phenotypes_tuple(phase[PhaseData.Type])) for phase in phases)
