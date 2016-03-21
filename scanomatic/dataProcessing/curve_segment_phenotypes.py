from scipy.ndimage import label
from scipy.stats import linregress
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from scanomatic.dataProcessing import growth_phenotypes
from scanomatic.dataProcessing import phenotyper
from enum import Enum


class CurvePhases(Enum):

    Multiple = -1
    Undetermined = 0
    Flat = 1
    Acceleration = 2
    Retardation = 3
    Impulse = 4


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
    ax.semilogy(times, curve, basey=2, color=colors[curve_color])
    ax.set_xlim(xmin=times[0], xmax=times[-1])

    return f


def new_phenotypes(phenotyper_object, plate, pos, impulse_threshold=0.75, flatline_threshold=0.02):

    loc = (np.subtract.outer(
        phenotyper_object.times,
        phenotyper_object.get_phenotype(phenotyper.Phenotypes.GenerationTimeWhen)[plate][pos]) ** 2).argmin(axis=0)

    dYdt = phenotyper_object.get_derivative(plate, pos)
    offset = (phenotyper_object.times.shape[0] - dYdt.shape[0]) / 2
    curve = phenotyper_object.smooth_growth_data[plate][pos]

    candidates = dYdt > dYdt[loc] * impulse_threshold
    candidates = signal.medfilt(candidates, 3).astype(bool)

    candidates, _ = label(candidates)
    candidates = candidates == candidates[loc]

    phases = np.ones_like(curve).astype(np.int) * -1

    where = np.where(candidates)[0]
    left = where[0]
    right = where[-1]

    padded_candidates = np.hstack(([False] * offset, candidates, [False] * offset))

    duration = phenotyper_object.times[right + offset] - phenotyper_object.times[left + offset]
    average_rate = 1 / linregress(phenotyper_object.times[padded_candidates],
                                  np.log2(curve[padded_candidates]))[0]
    impulse_yield = (np.log2(curve[right + offset]) - np.log2(curve[left + offset])) / \
        (np.log2(phenotyper_object.get_phenotype(growth_phenotypes.Phenotypes.CurveEndAverage)[plate][pos]) - \
         np.log2(phenotyper_object.get_phenotype(growth_phenotypes.Phenotypes.CurveBaseLine)[plate][pos]))

    ddYdt = signal.convolve(dYdt, [1, 0, -1], mode='valid')
    ddYdtSlice = slice(offset/2, -offset/2, None)
    candidates2 = candidates == False
    candidates2 = candidates2 & (np.abs(dYdt) > flatline_threshold)

    candidates2 = signal.medfilt(candidates2, 3).astype(bool)
    candidates2, _ = label(candidates2)

    try:
        acc_candidates = candidates2 == candidates2[(np.arange(candidates2.size) < left) &
                                                    (candidates2 > 0)].max()
    except ValueError:
        acc_candidates = np.zeros_like(candidates2).astype(bool)

    try:
        ret_candidates = candidates2 == candidates2[(np.arange(candidates2.size) > right) &
                                                    (candidates2 > 0)].min()
    except ValueError:
        ret_candidates = np.zeros_like(candidates2).astype(bool)


    return ({'GrowthImpulseDuration': duration,
             'GrowthImpulseGenerationsFraction': impulse_yield,
             'GrowthImpulseAverageRate': average_rate,
             'AccelerationPhaseMean': ddYdt[acc_candidates[ddYdtSlice]].mean() if acc_candidates.any() else np.nan,
             'RetartationPhaseMean': ddYdt[ret_candidates[ddYdtSlice]].mean() if ret_candidates.any() else np.nan},
            plot_segments(phenotyper_object.times, curve, phases))
