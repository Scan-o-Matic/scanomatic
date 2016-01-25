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


def iqr_mean_stable(data):

    if not isinstance(data, np.ma.masked_array):
        data = np.ma.masked_invalid(data)

    data = data[data.mask == False]
    threshold = np.floor(data.size * 0.25)
    data.sort()
    return data[threshold:-threshold].mean()


def quantiles_stable(data):

    if not isinstance(data, np.ma.masked_array):
        data = np.ma.masked_invalid(data)

    data = data[data.mask == False]
    threshold = np.floor(data.size * 0.25)
    data.sort()
    return data[threshold], data[-threshold]