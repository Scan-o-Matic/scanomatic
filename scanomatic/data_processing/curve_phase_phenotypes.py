from scipy.ndimage import label
from scipy.stats import linregress
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from scanomatic.data_processing import growth_phenotypes
from enum import Enum
import warnings
from functools import partial


class CurvePhases(Enum):

    Multiple = -1
    Undetermined = 0
    Flat = 1
    Acceleration = 2
    Retardation = 3
    Impulse = 4


class Thresholds(Enum):

    ImpulseExtension = 0
    ImpulseSlopeRequirement = 1
    FlatlineSlopRequirement = 2


class CurvePhasePhenotypes(Enum):

    Curvature = 0
    PopulationDoublingTime = 1
    Duration = 2
    FractionYield = 3
    Start = 4
    LinearModelSlope = 5
    LinearModelIntercept = 6
    AsymptoteAngle = 7
    AsymptoteIntersection = 8


class VectorPhenotypes(Enum):

    PhasesClassifications = 0
    PhasesPhenotypes = 1


def plot_segments(
        times, curve, phases, segment_alpha=0.3, f=None,
        colors={CurvePhases.Multiple: "#5f3275",
                CurvePhases.Flat: "#f9e812",
                CurvePhases.Acceleration: "#ea5207",
                CurvePhases.Impulse: "#99220c",
                CurvePhases.Retardation: "#c797c1"}):

    if f is None:
        f = plt.figure()

    ax = f.gca()

    for phase in CurvePhases:

        if phase == CurvePhases.Undetermined:
            continue

        labels, label_count = label(phases == phase.value)
        for id_label in range(1, label_count + 1):
            positions = np.where(labels == id_label)[0]
            left = positions[0]
            right = positions[-1]
            left = np.linspace(times[max(left - 1, 0)], times[left], 3)[1]
            right = np.linspace(times[min(curve.size - 1, right + 1)], times[right], 3)[1]
            ax.axvspan(left, right, color=colors[phase], alpha=segment_alpha)

    curve_color = CurvePhases(np.unique(phases)[0]) if np.unique(phases).size == 1 else CurvePhases.Multiple
    ax.semilogy(times, curve, basey=2, color=colors[curve_color], lw=2)
    ax.set_xlim(xmin=times[0], xmax=times[-1])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Population Size [cells]")

    return f


def _filter_find(vector, filter, func=np.max):

    return np.where((vector == func(vector[filter])) & filter)[0]


def _segment(dYdt, dYdtRanks, ddYdtSigns, phases, filter, offset,
             thresholds={Thresholds.ImpulseExtension: 0.75,
                         Thresholds.ImpulseSlopeRequirement: 0.1,
                         Thresholds.FlatlineSlopRequirement: 0.02}):

    if np.unique(phases[offset: -offset][filter]).size != 1:
        raise ValueError("Impossible to segment due to multiple phases {0} filter {1}".format(
            np.unique(phases[filter][offset: -offset]), filter
        ))

    # 1. Find segment's borders
    left, right = _locate_segment(filter)

    # 2. Find segment's maximum groth
    loc_max = _filter_find(dYdtRanks, filter)

    # 3. Further sementation requires existance of reliable growth impulse
    if dYdt[loc_max] < thresholds[Thresholds.ImpulseSlopeRequirement]:
        if filter.all():
            warnings.warn("No impulse detected, max rate is {0} (threshold {1}).".format(
                dYdt[loc_max], thresholds[Thresholds.ImpulseSlopeRequirement]))
        if left == 0:
            phases[:offset] = phases[offset]
        if right == phases.size:
            phases[-offset:] = phases[-offset - 1]
        return

    # 4. Locate impulse
    impulse_left, impulse_right = _locate_impulse(dYdt, loc_max, phases, filter, offset,
                                                  thresholds[Thresholds.ImpulseExtension])

    if impulse_left is None or impulse_right is None:
        if filter.all():
            warnings.warn("No impulse phase detected though max rate super-seeded threshold")
        if left == 0:
            phases[:offset] = phases[offset]
        if right == phases.size:
            phases[-offset:] = phases[-offset - 1]

        return

    # 5. Locate acceleration phase
    (accel_left, _), next_phase = _locate_acceleration(
        dYdt, ddYdtSigns, phases, left, impulse_left, offset,
        flatline_threshold=thresholds[Thresholds.FlatlineSlopRequirement])

    # 5b. If there's anything remaining, it is candidate flatline, but investigated for more impulses
    if left != accel_left:
        phases[left + offset: accel_left + 1 + offset] = next_phase.value
        _segment(dYdt, dYdtRanks, ddYdtSigns, phases, _get_filter(dYdt.size, left, accel_left), offset, thresholds)

    # 6. Locate retardation phase
    (_, retard_right), next_phase = _locate_retardation(
        dYdt, ddYdtSigns, phases, impulse_right, right, offset,
        flatline_threshold=thresholds[Thresholds.FlatlineSlopRequirement])

    # 6b. If there's anything remaining, it is candidate flatline, but investigated for more impulses
    if right != retard_right:
        phases[retard_right + offset: right + 1 + offset] = next_phase.value
        _segment(dYdt, dYdtRanks, ddYdtSigns, phases, _get_filter(dYdt.size, retard_right, right), offset, thresholds)

    # 7. Update phases edges
    phases[:offset] = phases[offset]
    phases[-offset:] = phases[-offset - 1]


def _locate_impulse(dYdt, loc, phases, filter, offset, extension_threshold):

    candidates = (dYdt > dYdt[loc] * extension_threshold) & filter
    candidates = signal.medfilt(candidates, 3).astype(bool)

    candidates, _ = label(candidates)
    phases[offset: -offset][candidates == candidates[loc]] = CurvePhases.Impulse.value
    return _locate_segment(candidates)


def _locate_segment(filt):

    where = np.where(filt)[0]
    if where.size > 0:
        return where[0], where[-1]
    else:
        return None, None


def _get_filter(size, left=None, right=None):

    filt = np.zeros(size).astype(bool)

    if left is None:
        left = 0
    if right is None:
        right = size - 1

    filt[left: right + 1] = True
    return filt


def _locate_acceleration(dYdt, ddYdtSigns, phases, left, right, offset, flatline_threshold):

    candidates = _get_filter(dYdt.size, left, right)
    candidates2 = candidates & (np.abs(dYdt) > flatline_threshold) & (ddYdtSigns == 1)

    candidates2 = signal.medfilt(candidates2, 3).astype(bool)
    candidates2, label_count = label(candidates2)

    if label_count:
        acc_candidates = candidates2 == label_count
        phases[offset: -offset][acc_candidates] = CurvePhases.Acceleration.value
        return _locate_segment(acc_candidates), CurvePhases.Flat
    else:
        phases[offset: -offset][candidates] = CurvePhases.Undetermined.value
        return (left, right), CurvePhases.Undetermined


def _locate_retardation(dYdt, ddYdtSigns, phases, left, right, offset, flatline_threshold):

    candidates = _get_filter(dYdt.size, left, right)
    candidates2 = candidates & (np.abs(dYdt) > flatline_threshold) & (ddYdtSigns == -1)

    candidates2 = signal.medfilt(candidates2, 3).astype(bool)
    candidates2, label_count = label(candidates2)

    if label_count:
        ret_cantidates = candidates2 == 1
        phases[offset: -offset][ret_cantidates] = CurvePhases.Retardation.value
        return _locate_segment(ret_cantidates), CurvePhases.Flat
    else:
        phases[offset: -offset][candidates] = CurvePhases.Undetermined.value
        return (left, right), CurvePhases.Undetermined


def _phenotype_phases(curve, derivative, phases, times, doublings):

    derivative_offset = (times.shape[0] - derivative.shape[0]) / 2
    phenotypes = []

    for phase in CurvePhases:

        labels, label_count = label(phases == phase.value)
        for id_label in range(1, label_count + 1):

            if phase == CurvePhases.Undetermined or phase == CurvePhases.Multiple:
                phenotypes.append((phase, None))
                continue

            filt = labels == id_label
            left, right = _locate_segment(filt)

            phase_phenotypes = {}

            if phase == CurvePhases.Acceleration or phase == CurvePhases.Retardation:
                # A. For non-linear phases use the X^2 coefficient as curvature measure
                phase_phenotypes[CurvePhasePhenotypes.Curvature] = np.polyfit(times[filt], np.log2(curve[filt]), 2)[0]

                a1 = np.array((times[left], np.log2(curve[left])))
                a2 = np.array((times[right], np.log2(curve[right])))
                k1 = derivative[left - derivative_offset]
                k2 = derivative[right - derivative_offset]
                m1 = a1[1] - k1 * a1[0]
                m2 = a2[1] - k2 * a2[1]
                i_x = (m2 - m1) / (k1 - k2)
                i = np.array((i_x, k1 * i_x + m1))
                a1 -= i
                a2 -= i
                phase_phenotypes[CurvePhasePhenotypes.AsymptoteIntersection] = (i_x - times[left]) / (times[right] - times[left])
                phase_phenotypes[CurvePhasePhenotypes.AsymptoteAngle] = np.arccos(
                    np.dot(a1, a2) / (np.sqrt(np.dot(a1, a1)) * np.sqrt(np.dot(a2, a2))))
            else:
                # B. For linear phases get the doubling time
                slope, intercept, _, _, _ = linregress(times[filt], np.log2(curve[filt]))
                phase_phenotypes[CurvePhasePhenotypes.PopulationDoublingTime] = 1 / slope
                phase_phenotypes[CurvePhasePhenotypes.LinearModelSlope] = slope
                phase_phenotypes[CurvePhasePhenotypes.LinearModelIntercept] = intercept

            # C. Get duration
            phase_phenotypes[CurvePhasePhenotypes.Duration] = times[right] - times[left]

            # D. Get fraction of doublings
            phase_phenotypes[CurvePhasePhenotypes.FractionYield] = (np.log2(curve[right]) - np.log2(curve[left])) / \
                                                                   doublings


            # E. Get start of phase
            phase_phenotypes[CurvePhasePhenotypes.Start] = times[left]

            phenotypes.append((phase, phase_phenotypes))

    # Phenotypes sorted on phase start rather than type of phase
    return sorted(phenotypes, key=lambda (t, p): p[CurvePhasePhenotypes.Start] if p is not None else 9999)


def phase_phenotypes(
        phenotyper_object, plate, pos, segment_alpha=0.75, f=None,
        thresholds={Thresholds.ImpulseExtension: 0.75,
                    Thresholds.ImpulseSlopeRequirement: 0.1,
                    Thresholds.FlatlineSlopRequirement: 0.02},
        experiment_doublings=None):

    curve = phenotyper_object.smooth_growth_data[plate][pos]
    dYdt = phenotyper_object.get_derivative(plate, pos)
    offset = (phenotyper_object.times.shape[0] - dYdt.shape[0]) / 2
    dYdtRanks = dYdt.argsort().argsort()
    ddYdt = signal.convolve(dYdt, [1, 0, -1], mode='valid')
    ddYdtSigns = np.hstack(([0], np.sign(ddYdt), [0]))
    phases = np.ones_like(curve).astype(np.int) * 0
    span = _get_filter(dYdt.size)

    _segment(dYdt, dYdtRanks, ddYdtSigns, phases, filter=span, offset=offset, thresholds=thresholds)

    if experiment_doublings is None:
        experiment_doublings = (np.log2(phenotyper_object.get_phenotype(
            growth_phenotypes.Phenotypes.ExperimentEndAverage)[plate][pos]) -
                                np.log2(phenotyper_object.get_phenotype(
                                    growth_phenotypes.Phenotypes.ExperimentBaseLine)[plate][pos]))

    return phases,\
           _phenotype_phases(curve, dYdt, phases, phenotyper_object.times, experiment_doublings),\
           None if f is False else plot_segments(phenotyper_object.times, curve, phases, segment_alpha=segment_alpha,
                                                 f=f)


def phase_selector_critera_filter(phases, criteria, func=max):

    val = func(phase[criteria] for phase in phases)
    return tuple(phase for phase in phases if phase[criteria] == val)[0]


def filter_plate_custom_filter(
        plate,
        phase=CurvePhases.Acceleration,
        measure=CurvePhasePhenotypes.Curvature,
        phases_requirement=lambda phases: len(phases) == 1,
        phase_selector=lambda phases: phases[0]):

    def f(v):
        phases = tuple(d for t, d in v if t == phase)
        if phases_requirement(phases):
            return phase_selector(phases)[measure]
        return np.nan

    return np.ma.masked_invalid(np.frompyfunc(f, 1, 1)(plate).astype(np.float))


class CurvePhaseMetaPhenotypes(Enum):

    MajorImpulseYieldContribution = 0
    BimodalGrowthFirstImpulseDoubingTime = 1
    BimodalGrowthSecondImpulseDoubingTime = 2
    InitialLag = 3


def filter_plate(plate, meta_phenotype):

    if meta_phenotype == CurvePhaseMetaPhenotypes.MajorImpulseYieldContribution:

        selector = partial(phase_selector_critera_filter, criteria=CurvePhasePhenotypes.FractionYield)
        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.FractionYield,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=selector)

    elif (meta_phenotype == CurvePhaseMetaPhenotypes.BimodalGrowthFirstImpulseDoubingTime or
          meta_phenotype == CurvePhaseMetaPhenotypes.BimodalGrowthSecondImpulseDoubingTime):

        return filter_plate_custom_filter(
            plate,
            phase=CurvePhases.Impulse,
            measure=CurvePhasePhenotypes.PopulationDoublingTime,
            phases_requirement=lambda phases: len(phases) == 2,
            phase_selector=lambda phases: phases[
                0 if meta_phenotype == CurvePhaseMetaPhenotypes.BimodalGrowthFirstImpulseDoubingTime else 1]
        )

    elif meta_phenotype == CurvePhaseMetaPhenotypes.InitialLag:

        lag_slope = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        lag_intercept = filter_plate_custom_filter(
            plate, phase=CurvePhases.Flat, measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        impulse_slope = filter_plate_custom_filter(
            plate, phase=CurvePhases.Impulse, measure=CurvePhasePhenotypes.LinearModelSlope,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        impulse_intercept = filter_plate_custom_filter(
            plate, phase=CurvePhases.Impulse, measure=CurvePhasePhenotypes.LinearModelIntercept,
            phases_requirement=lambda phases: len(phases) > 0,
            phase_selector=lambda phases: phases[0])

        return (impulse_intercept - lag_intercept) / (lag_slope - impulse_slope)