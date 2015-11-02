__author__ = 'martin'

from scipy.stats.mstats import mquantiles
import numpy as np


def iqr_mean(data, *args, **kwargs):
    quantiles = mquantiles(data, prob=(0.25, 0.75))
    if quantiles.any():
        val = np.ma.masked_outside(data, *quantiles).mean(*args, **kwargs)
        if isinstance(val, np.ma.MaskedArray):
            return  val.filled(np.nan)
        return val
    return None