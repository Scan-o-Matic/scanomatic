import numpy as np
from enum import Enum

from scanomatic.data_processing import growth_phenotypes
from scanomatic.io.logger import Logger
from scanomatic.data_processing.phases.analysis import CurvePhasePhenotypes
from scanomatic.data_processing.phases.segmentation import CurvePhases, is_detected_non_linear

_l = Logger("Curve Phase Meta Phenotyping")


class CurvePhaseMetaPhenotypes(Enum):
    """Phenotypes of an entire growth-log2_curve based on the phase segmentation.

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
    FirstMinorImpulseYieldContribution = 1
    MajorImpulseAveragePopulationDoublingTime = 5
    FirstMinorImpulseAveragePopulationDoublingTime = 6
    MajorImpulseFlankAsymmetry = 8

    InitialAccelerationAsymptoteAngle = 10
    FinalRetardationAsymptoteAngle = 11
    InitialAccelerationAsymptoteIntersect = 15
    FinalRetardationAsymptoteIntersect = 16

    InitialLag = 20
    InitialLagAlternativeModel = 22

    Modalities = 25
    ModalitiesAlternativeModel = 27

    Collapses = 26


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

    return sum(1 for t, d in phase_vector if t != CurvePhases.Undetermined)

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
            p_data[CurvePhasePhenotypes.FractionYield] if
            p_data is not None and p_data[CurvePhasePhenotypes.FractionYield] else -np.inf for p_type, p_data in phases
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
        impulses_phase = _get_phase_id(plate, CurvePhases.Flat, CurvePhases.Impulse)

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
        if v.ndim == 1 and v.shape[0] and (vshape is None or v.shape == vshape):
            if vshape is None:
                vshape = v.shape
            data.append(v)
    return np.ma.array(data)


def get_phase_assignment_frequencies(phenotypes, plate):

    data = get_phase_assignment_data(phenotypes, plate)
    min_length = data.max() + 1
    bin_counts = [np.bincount(data[..., i], minlength=min_length) for i in range(data.shape[1])]
    return np.array(bin_counts)


def _get_index_array(shape):

    m = np.mgrid[:shape[0], :shape[1]]

    l = zip(*(v.ravel() for v in m))
    a2 = np.empty(m.shape[1:], dtype=np.object)
    a2.ravel()[:] = l
    return a2


def get_phase_phenotypes_aligned(phenotypes, plate):

    p = phenotypes._vector_phenotypes[plate][VectorPhenotypes.PhasesPhenotypes]
    filt = phenotypes.get_curves_filter_compacted(plate)
    coords = _get_index_array(p.shape)

    p = p[filt == np.False_]
    coords = coords[filt == np.False_]

    major_idx = np.ma.masked_invalid(_np_ma_get_major_impulse_indices(p).astype(np.float))
    p = p[major_idx.mask == np.False_]
    coords = coords[major_idx.mask == np.False_]

    l = _np_phase_counter(p)
    id_most_left_phases = major_idx.argmax()
    id_most_right_phases = (l - major_idx).argmax()

    return id_most_left_phases, id_most_right_phases