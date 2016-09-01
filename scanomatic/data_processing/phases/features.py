import numpy as np
from enum import Enum

from scanomatic.data_processing import growth_phenotypes

from scanomatic.data_processing.phases.analysis import CurvePhasePhenotypes
from scanomatic.data_processing.phases.segmentation import CurvePhases, is_detected_non_linear

# TODO: CurvePhaseMetaPhenotypes.MajorImpulseFlankAsymmetry could consider
# flanking flats too, calculating impulse angle to flat.

# TODO: Consider using cached pre calculations and using one time np.frompyfunc


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
    MajorImpulseFlankAsymmetry = 8

    InitialAccelerationAsymptoteAngle = 10
    FinalRetardationAsymptoteAngle = 11
    InitialAccelerationAsymptoteIntersect = 15
    FinalRetardationAsymptoteIntersect = 16

    InitialLag = 20
    InitialLagAlternativeModel = 22

    ExperimentDoublings = 21

    Modalities = 25
    ModalitiesAlternativeModel = 27

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


def _phase_finder(phase_vector, phase):

    if phase_vector:
        return tuple(i for i, (p_type, p_data) in enumerate(phase_vector) if p_type == phase)
    return tuple()

# REGION: Phase counters


def _py_impulse_counter(phase_vector):
    if phase_vector:
        return sum(1 for phase in phase_vector if phase[0] == CurvePhases.Impulse)
    return -np.inf

_np_impulse_counter = np.frompyfunc(_py_impulse_counter, 1, 1)


def _py_inner_impulse_counter(phase_vector):

    if phase_vector:
        acc = _phase_finder(phase_vector, CurvePhases.GrowthAcceleration)
        if not acc:
            return -np.inf
        ret = _phase_finder(phase_vector, CurvePhases.GrowthRetardation)
        if not ret:
            return -np.inf
        return _py_impulse_counter(phase_vector[acc[0]: ret[-1]])

    return -np.inf

_np_inner_impulse_counter = np.frompyfunc(_py_inner_impulse_counter, 1, 1)


def _py_collapse_counter(phase_vector):
    if phase_vector:
        return sum(1 for phase in phase_vector if phase[0] == CurvePhases.Collapse)
    return -np.inf

_np_collapse_counter = np.frompyfunc(_py_collapse_counter, 1, 1)

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

    Returns: 2D numpy.ma.masked_array with indices of the major
        growth impulses in the vectors.
    """

    sort_order = np.argsort(tuple(
        p_data[CurvePhasePhenotypes.FractionYield] if
        p_data[CurvePhasePhenotypes.FractionYield] else -np.inf for p_type, p_data in phases))

    impulses = np.array(tuple(
        (i, v) for i, v in enumerate(sort_order) if
        phases[i][VectorPhenotypes.PhasesClassifications.value] == CurvePhases.Impulse))

    if impulses.any():
        return impulses[np.argmax(impulses[:, -1])][0]
    return -1

_np_get_major_impulse_for_plate = np.frompyfunc(_py_get_major_impulse_for_plate, 1, 1)


def _np_ma_get_major_impulse_indices(phases):

    return np.ma.masked_less(_np_get_major_impulse_for_plate(phases), 0)

# END REGION: Major pulse index


def _py_get_flanking_angle_relation(phases, major_impulse_index):

    def _flank_angle(flank, impulse):

        if flank is None:

            return np.arctan(impulse[VectorPhenotypes.PhasesPhenotypes][CurvePhasePhenotypes.LinearModelSlope])

        elif flank[VectorPhenotypes.PhasesClassifications] is CurvePhases.Flat:

            return np.pi - np.abs(
                np.arctan2(1, impulse[VectorPhenotypes.PhasesPhenotypes][CurvePhasePhenotypes.LinearModelSlope]) -
                np.arctan2(1, flank[VectorPhenotypes.PhasesPhenotypes][CurvePhasePhenotypes.LinearModelSlope]))

        elif flank[VectorPhenotypes.PhasesClassifications] in (
                CurvePhases.CollapseAcceleration, CurvePhases.GrowthAcceleration,
                CurvePhases.CollapseRetardation, CurvePhases.GrowthRetardation):

            return flank[VectorPhenotypes.PhasesPhenotypes][CurvePhasePhenotypes.AsymptoteAngle]

        else:
            return np.inf

    if not major_impulse_index:
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

        return np.ma.masked_invalid(_np_impulse_counter(plate).astype(np.float))

    elif meta_phenotype == CurvePhaseMetaPhenotypes.ModalitiesAlternativeModel:

        return np.ma.masked_invalid(np.frompyfunc(_np_inner_impulse_counter(plate)).astype(np.float))

    elif meta_phenotype == CurvePhaseMetaPhenotypes.Collapses:

        return np.ma.masked_invalid(_np_collapse_counter(plate)).astype(np.float)

    elif meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseFlankAsymmetry:

        return _np_get_flanking_angle_relation(plate, _np_ma_get_major_impulse_indices(plate))

    else:

        return np.ma.masked_invalid(np.ones_like(plate.shape) * np.nan)
