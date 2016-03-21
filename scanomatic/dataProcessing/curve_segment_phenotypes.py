from scipy.ndimage import label
from scipy.stats import linregress
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from scanomatic.dataProcessing import growth_phenotypes
from enum import Enum
import warnings


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
        return

    # 4. Locate impulse
    impulse_left, impulse_right = _locate_impulse(dYdt, loc_max, phases, filter, offset,
                                                  thresholds[Thresholds.ImpulseExtension])

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


def _locate_segment(filter):

    where = np.where(filter)[0]
    return where[0], where[-1]


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
        acc_candidates = candidates2 == label_count
        phases[offset: -offset][acc_candidates] = CurvePhases.Retardation.value
        return _locate_segment(acc_candidates), CurvePhases.Flat
    else:
        phases[offset: -offset][candidates] = CurvePhases.Undetermined.value
        return (left, right), CurvePhases.Undetermined


def _phenotype_phases(curve, phases, phenotyper_object, plate, pos):

    times = phenotyper_object.times
    phenotypes = []
    doublings = (np.log2(phenotyper_object.get_phenotype(growth_phenotypes.Phenotypes.CurveEndAverage)[plate][pos]) - \
                 np.log2(phenotyper_object.get_phenotype(growth_phenotypes.Phenotypes.CurveBaseLine)[plate][pos]))

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
            else:
                # B. For linear phases get the doubling time
                phase_phenotypes[CurvePhasePhenotypes.PopulationDoublingTime] = 1 / linregress(times[filt], np.log2(curve[filt]))[0]

            # C. Get duration
            phase_phenotypes[CurvePhasePhenotypes.Duration] = times[right] - times[left]

            # D. Get fraction of doublings
            phase_phenotypes[CurvePhasePhenotypes.FractionYield] = (np.log2(curve[right]) - np.log2(curve[left])) / \
                                                                   doublings


            # E. Get start of phase
            phase_phenotypes[CurvePhasePhenotypes.Start] = times[left]

            phenotypes.append((phase, phase_phenotypes))

    # Return phenotypes sorted on phase start rather than type of phase
    return sorted(phenotypes, key=lambda (t, p): p[CurvePhasePhenotypes.Start])


def phase_phenotypes(
        phenotyper_object, plate, pos, segment_alpha=0.75, f=None,
        thresholds={Thresholds.ImpulseExtension: 0.75,
                    Thresholds.ImpulseSlopeRequirement: 0.1,
                    Thresholds.FlatlineSlopRequirement: 0.02}):

    curve = phenotyper_object.smooth_growth_data[plate][pos]
    dYdt = phenotyper_object.get_derivative(plate, pos)
    offset = (phenotyper_object.times.shape[0] - dYdt.shape[0]) / 2
    dYdtRanks = dYdt.argsort().argsort()
    ddYdt = signal.convolve(dYdt, [1, 0, -1], mode='valid')
    ddYdtSigns = np.hstack(([0], np.sign(ddYdt), [0]))
    phases = np.ones_like(curve).astype(np.int) * 0
    span = _get_filter(dYdt.size)

    _segment(dYdt, dYdtRanks, ddYdtSigns, phases, filter=span, offset=offset, thresholds=thresholds)

    return phases,\
           _phenotype_phases(curve, phases, phenotyper_object, plate, pos),\
           None if f is False else plot_segments(phenotyper_object.times, curve, phases, segment_alpha=segment_alpha,
                                                 f=f)
