from scipy.ndimage import label
from scipy.stats import linregress
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from scanomatic.dataProcessing import growth_phenotypes
from scanomatic.dataProcessing import phenotyper


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

    f = plt.figure()
    ax = f.gca()
    ax.semilogy(curve, basey=2)
    ax.axvspan(left + 2, right + 3, color='m', alpha=.2)
    if acc_candidates.any():
        ax.axvspan(
            np.where(acc_candidates)[0][0] + 2,
            np.where(acc_candidates)[0][-1] + 3, color='c', alpha=0.2)
    if ret_candidates.any():
        ax.axvspan(
            np.where(ret_candidates)[0][0] + 2,
            np.where(ret_candidates)[0][-1] + 3, color='g', alpha=0.2)

    return ({'GrowthImpulseDuration': duration,
             'GrowthImpulseGenerationsFraction': impulse_yield,
             'GrowthImpulseAverageRate': average_rate,
             'AccelerationPhaseMean': ddYdt[acc_candidates[ddYdtSlice]].mean() if acc_candidates.any() else np.nan,
             'RetartationPhaseMean': ddYdt[ret_candidates[ddYdtSlice]].mean() if ret_candidates.any() else np.nan},
            f)
