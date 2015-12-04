from enum import Enum
import numpy as np
from scipy.optimize import leastsq
from itertools import izip
from scipy.stats import linregress


def _linreg_helper(X, Y):
    return linregress(X, Y)[0::4]


def get_preprocessed_data_for_phenotypes(curve, curve_strided, flat_times, times_strided, index_for_48h,
                                         position_offset):

    linreg_values = []
    curve_logged = np.log2(curve)

    for times, value_segment in izip(times_strided, curve_strided):

        linreg_values.append(_linreg_helper(times, value_segment))

    derivative_values, derivative_errors = np.array(linreg_values).T

    return {
        'curve_smooth_growth_data': np.ma.masked_invalid(curve),
        'index48h': index_for_48h,
        'chapman_richards_fit': CalculateFitRSquare(flat_times, curve_logged),
        'derivative_values': derivative_values,
        'derivative_errors': derivative_errors,
        'linregress_extent': position_offset,
        'flat_times': flat_times}


def initial_value(curve_smooth_growth_data, *args, **kwargs):
    return curve_smooth_growth_data[0]


def curve_first_two_average(curve_smooth_growth_data, *args, **kwargs):
    return curve_smooth_growth_data[:2].mean()


def curve_baseline(curve_smooth_growth_data, *args, **kwargs):
    return curve_smooth_growth_data[:3].mean()


def curve_low_point(curve_smooth_growth_data, *args, **kwargs):
    return curve_smooth_growth_data[:3].min()


def curve_end_average(curve_smooth_growth_data, *args, **kwargs):
    return curve_smooth_growth_data[3:].mean()


def growth_yield(curve_smooth_growth_data, *args, **kwargs):
    return curve_end_average(curve_smooth_growth_data) - curve_baseline(curve_smooth_growth_data)


def growth_48h(curve_smooth_growth_data, index48h, *args, **kwargs):
    return curve_smooth_growth_data[index48h]


def ChapmanRichards4ParameterExtendedCurve(X, b0, b1, b2, b3, D):
    """Reterns a Chapman-Ritchards 4 parameter curve exteneded with a
    Y-axis offset D parameter.

    ''Note: The parameters b0, b1, b2 and b3 have been transposed so
    that they stay within the allowed bounds of the model

    Args:

        X (np.array):   The X-data

        b0 (float): The first parameter. To ensure that it stays within
                    the allowed bounds b0 > 0, the input b0 is
                    transposed using ``np.power(np.e, b0)``.

        b1 (float): The second parameter. The bounds are
                    1 - b3 < b1 < 1 and thus it is scaled as follows::

                        ``np.power(np.e, b1) / (np.power(np.e, b1) + 1) *
                        b3 + (1 - b3)``

                    Where ``b3`` referes to the transformed version.


        b2 (float): The third parameter, has same bounds and scaling as
                    the first

        b3 (float): The fourth parameter, has bounds 0 < b3 < 1, thus
                    scaling is done with::

                        ``np.power(np.e, b3) / (np.power(np.e, b3) + 1)``

        D (float):  Any real number, used as the offset of the curve,
                    no transformation applied.

    Returns:

        np.array.       An array of matching size as X with the
                        Chapman-Ritchards extended curve for the
                        parameter set.

    """

    # Enusuring parameters stay within the allowed bounds
    b0 = np.power(np.e, b0)
    b2 = np.power(np.e, b2)
    v = np.power(np.e, b3)
    b3 = v / (v + 1.0)
    v = np.power(np.e, b1)
    b1 = v / (v + 1.0) * b3 + (1 - b3)

    return D + b0 * np.power(1.0 - b1 * np.exp(-b2 * X), 1.0 / (1.0 - b3))


def CalculateFitRSquare(X, Y, p0=np.array([1.64, -0.1, -2.46, 0.1, 15.18],dtype=np.float)):
    """X and Y must be 1D, Y must be log2"""

    p = leastsq(RCResiduals, p0, args=(X, Y))[0]
    Yhat = ChapmanRichards4ParameterExtendedCurve(
        X, *p)
    return (1.0 - np.square(Yhat - Y).sum() /
        np.square(Yhat - Y[np.isfinite(Y)].mean()).sum()), p


def RCResiduals(crParams, X, Y):

    return Y - ChapmanRichards4ParameterExtendedCurve(X, *crParams)


def generation_time(derivative_values, index, **kwargs):
    return 1.0 / derivative_values[index]


def generation_time_error(derivative_error, index, **kwargs):
    return derivative_error[index]


def generation_time_when(flat_times, index):
    return flat_times[index]


def population_size_at_generation_time(curve_smooth_growth_data, index, linregress_extent, **kwargs):

    return np.median(
        curve_smooth_growth_data[
            max(0, index - linregress_extent):
            min(index + linregress_extent + 1, curve_smooth_growth_data.size)])


def growth_lag(index, flat_times, derivative_values, **kwargs):

    growth_delta = population_size_at_generation_time(index=index, **kwargs) - curve_baseline(**kwargs)

    if growth_delta > 0:

        return np.interp(max(0.0, -growth_delta / derivative_values[index]), np.arange(flat_times.size), flat_times)

    return np.nan

#
#
# Enum helpers
#

_generation_time_indices = None
_kwargs = None


def _get_generation_time_index(kwargs, rank):
    global _kwargs, _generation_time_indices
    if kwargs is not _kwargs:
        _kwargs = kwargs
        _generation_time_indices = np.argsort(kwargs['derivative_values'])

    return _generation_time_indices[rank]


class Phenotypes(Enum):

    InitialValue = 12
    CurveFirstTwoAverage = 13
    CurveBaseLine = 14
    CurveLowPoint = 15
    CurveEndAverage = 16

    CurveGrowthYield = 17
    GrowthLag = 18

    GenerationTime48h = 19
    ColonySize48h = 20

    GenerationTime = 0
    GenerationTimeStErrOfEstimate = 1
    GenerationTimeScanIndex = 2
    GenerationTimePopulationSize = 21

    GenerationTime2 = 3
    GenerationTime2StErrOfEstimate = 4
    GenerationTime2ScanIndex = 5

    ChapmanRichardsFit = 6
    ChapmanRichardsParam1 = 7
    ChapmanRichardsParam2 = 8
    ChapmanRichardsParam3 = 9
    ChapmanRichardsParam4 = 10
    ChapmanRichardsParamXtra = 11

    def __call__(self, **kwargs):

        if self is Phenotypes.InitialValue:
            return initial_value(**kwargs)

        elif self is Phenotypes.CurveBaseLine:
            return curve_baseline(**kwargs)

        elif self is Phenotypes.CurveEndAverage:
            return curve_end_average(**kwargs)

        elif self is Phenotypes.CurveFirstTwoAverage:
            return curve_first_two_average(**kwargs)

        elif self is Phenotypes.ColonySize48h:
            return growth_48h(**kwargs)

        elif self is Phenotypes.CurveGrowthYield:
            return growth_yield(**kwargs)

        elif self is Phenotypes.CurveLowPoint:
            return curve_low_point(**kwargs)

        elif self is Phenotypes.ChapmanRichardsFit:
            return kwargs['chapman_richards_fit'][0]

        elif self is Phenotypes.ChapmanRichardsParam1:
            return kwargs['chapman_richards_fit'][1][0]

        elif self is Phenotypes.ChapmanRichardsParam2:
            return kwargs['chapman_richards_fit'][1][1]

        elif self is Phenotypes.ChapmanRichardsParam3:
            return kwargs['chapman_richards_fit'][1][2]

        elif self is Phenotypes.ChapmanRichardsParam4:
            return kwargs['chapman_richards_fit'][1][3]

        elif self is Phenotypes.ChapmanRichardsParamXtra:
            return kwargs['chapman_richards_fit'][1][4]

        elif self is Phenotypes.GenerationTime:
            return generation_time(index=_generation_time_indices(kwargs, 0), **kwargs)

        elif self is Phenotypes.GenerationTime2:
            return generation_time(index=_generation_time_indices(kwargs, 1), **kwargs)

        elif self is Phenotypes.GenerationTimeScanIndex:
            return generation_time_when(_generation_time_indices(kwargs, 0), **kwargs)

        elif self is Phenotypes.GenerationTime2ScanIndex:
            return generation_time_when(_generation_time_indices(kwargs, 1), **kwargs)

        elif self is Phenotypes.GenerationTimeScanIndex:
            return generation_time_error(index=_generation_time_indices(kwargs, 0), **kwargs)

        elif self is Phenotypes.GenerationTime2StErrOfEstimate:
            return generation_time_error(index=_generation_time_indices(kwargs, 1), **kwargs)

        elif self is Phenotypes.GenerationTimePopulationSize:
            return population_size_at_generation_time(index=_generation_time_indices(kwargs, 0), **kwargs)

        elif self is Phenotypes.GrowthLag:
            return growth_lag(index=_generation_time_indices(kwargs, 0), **kwargs)