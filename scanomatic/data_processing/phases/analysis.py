import numpy as np
from enum import Enum
from scipy.ndimage import label
from scipy.stats import linregress

from scanomatic.data_processing import growth_phenotypes

from scanomatic.data_processing.phases.segmentation import CurvePhases, DEFAULT_THRESHOLDS, segment, \
    get_data_needed_for_segmentation, is_detected_non_linear, is_detected_linear, is_undetermined


class CurvePhasePhenotypes(Enum):
    """Phenotypes for individual log2_curve phases.

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


def _phenotype_phases(model, doublings):

    phenotypes = []

    # noinspection PyTypeChecker
    for phase in CurvePhases:

        labels, label_count = label(model.phases == phase.value)
        for id_label in range(1, label_count + 1):

            if is_undetermined(phase):
                phenotypes.append((phase, None))
                continue

            filt = labels == id_label
            left, right = _locate_segment(filt)
            time_right = model.times[right - 1]
            time_left = model.times[left]
            current_phase_phenotypes = {}

            if is_detected_non_linear(phase):
                # A. For non-linear phases use the X^2 coefficient as curvature measure

                # TODO: Verify that values fall within the defined range of 0.5pi and pi

                k1 = model.dydt[max(0, left - model.offset)]
                k2 = model.dydt[right - 1 - model.offset]
                m1 = model.log2_curve[left] - k1 * time_left
                m2 = model.log2_curve[right - 1] - k2 * time_right
                i_x = (m2 - m1) / (k1 - k2)
                current_phase_phenotypes[CurvePhasePhenotypes.AsymptoteIntersection] = \
                    (i_x - time_left) / (time_right - time_left)
                current_phase_phenotypes[CurvePhasePhenotypes.AsymptoteAngle] = \
                    np.pi - np.abs(np.arctan2(k1, 1) - np.arctan2(k2, 1))

            elif is_detected_linear(phase):
                # B. For linear phases get the doubling time
                slope, intercept, _, _, _ = linregress(model.times[filt], model.log2_curve[filt])
                current_phase_phenotypes[CurvePhasePhenotypes.PopulationDoublingTime] = 1 / slope
                current_phase_phenotypes[CurvePhasePhenotypes.LinearModelSlope] = slope
                current_phase_phenotypes[CurvePhasePhenotypes.LinearModelIntercept] = intercept

            # C. Get duration
            current_phase_phenotypes[CurvePhasePhenotypes.Duration] = time_right - time_left

            # D. Get fraction of doublings
            current_phase_phenotypes[CurvePhasePhenotypes.FractionYield] = \
                (model.log2_curve[right - 1] - model.log2_curve[left]) / doublings

            # E. Get start of phase
            current_phase_phenotypes[CurvePhasePhenotypes.Start] = time_left

            phenotypes.append((phase, current_phase_phenotypes))

    # Phenotypes sorted on phase start rather than type of phase
    return sorted(phenotypes, key=lambda (t, p): p[CurvePhasePhenotypes.Start] if p is not None else 9999)


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
        raise ValueError("Filter is not homogeneous, contains {0} segments ({1})".format(n, labels.tolist()))
    else:
        return None, None


def get_phase_analysis(phenotyper_object, plate, pos, thresholds=None, experiment_doublings=None):

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    model = get_data_needed_for_segmentation(phenotyper_object, plate, pos, thresholds)

    for _ in segment(model, thresholds):

        pass

    if experiment_doublings is None:

        experiment_doublings = phenotyper_object.get_phenotype(
            growth_phenotypes.Phenotypes.ExperimentPopulationDoublings)[plate][pos]

    # TODO: ensure it isn't unintentionally smoothed dydt that is uses for values, good for location though
    return model.phases, _phenotype_phases(model, experiment_doublings)
