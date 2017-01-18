from enum import Enum
import numpy as np
from scipy.optimize import leastsq
from itertools import izip
from scipy.stats import linregress
from scanomatic.io.logger import Logger

_logger = Logger("Growth Phenotypes")


def _linreg_helper(X, Y):
    return linregress(X, Y)[0::4]


def get_derivative(curve_strided, times_strided):

    linreg_values = []

    log2_strided_curve = np.log2(curve_strided)
    filters = np.isfinite(log2_strided_curve)
    min_size = curve_strided.shape[-1] - 1

    for times, value_segment, filt in izip(times_strided, log2_strided_curve, filters):

        if filt.sum() >= min_size:
            linreg_values.append(_linreg_helper(times[filt], value_segment[filt]))
        else:
            linreg_values.append((np.nan, np.nan))

    derivative_values_log2, derivative_errors = np.array(linreg_values).T

    return derivative_values_log2, derivative_errors


def get_preprocessed_data_for_phenotypes(curve, curve_strided, flat_times, times_strided, index_for_48h,
                                         position_offset):

    curve_logged = np.log2(curve)
    derivative_values_log2, derivative_errors = get_derivative(curve_strided, times_strided)

    return {
        'curve_smooth_growth_data': np.ma.masked_invalid(curve),
        'index48h': index_for_48h,
        'chapman_richards_fit': get_fit_r_square(flat_times, curve_logged),
        'derivative_values_log2': np.ma.masked_invalid(derivative_values_log2),
        'derivative_errors': np.ma.masked_invalid(derivative_errors),
        'linregress_extent': position_offset,
        'flat_times': flat_times}


def initial_value(curve_smooth_growth_data, *args, **kwargs):
    return curve_smooth_growth_data[0]


def curve_first_two_average(curve_smooth_growth_data, *args, **kwargs):
    if curve_smooth_growth_data[:2].any():
        return curve_smooth_growth_data[:2].mean()
    else:
        return np.nan


def curve_baseline(curve_smooth_growth_data, *args, **kwargs):
    if curve_smooth_growth_data[:3].any():
        return curve_smooth_growth_data[:3].mean()
    else:
        return np.nan


def curve_low_point(curve_smooth_growth_data, *args, **kwargs):
    if curve_smooth_growth_data.any():
        return np.ma.masked_invalid(np.convolve(curve_smooth_growth_data, np.ones(3) / 3., mode='valid')).min()
    else:
        return np.nan


def curve_low_point_time(curve_smooth_growth_data, flat_times, *args, **kwargs):
    # TODO: If a keeper make the convoloution be precalc and not done twice (se above func)
    try:
        return flat_times[
            np.ma.masked_invalid(np.convolve(curve_smooth_growth_data, np.ones(3) / 3., mode='valid')).argmin() + 1]
    except ValueError:
        return np.nan


def curve_end_average(curve_smooth_growth_data, *args, **kwargs):
    if curve_smooth_growth_data[-3:].any():
        return curve_smooth_growth_data[-3:].mean()
    else:
        return np.nan


def curve_monotonicity(curve_smooth_growth_data, *args, **kwargs):
    return (np.diff(curve_smooth_growth_data[curve_smooth_growth_data.mask == np.False_]) > 0).astype(float).sum() / \
           (curve_smooth_growth_data.size - 1)


def growth_yield(curve_smooth_growth_data, *args, **kwargs):
    return curve_end_average(curve_smooth_growth_data) - curve_baseline(curve_smooth_growth_data)


def growth_curve_doublings(curve_smooth_growth_data, *args, **kwargs):
    return np.log2(curve_end_average(curve_smooth_growth_data)) - np.log2(curve_baseline(curve_smooth_growth_data))


def residual_growth(curve_smooth_growth_data, *args, **kwargs):
    return curve_end_average(curve_smooth_growth_data) - \
           population_size_at_generation_time(
               curve_smooth_growth_data=curve_smooth_growth_data,
               index=_get_generation_time_index(kwargs['derivative_values_log2'], 0),
               **kwargs)


def residual_growth_as_population_doublings(curve_smooth_growth_data, *args, **kwargs):
    return np.log2(curve_end_average(curve_smooth_growth_data)) - \
           np.log2(population_size_at_generation_time(
               curve_smooth_growth_data=curve_smooth_growth_data,
               index=_get_generation_time_index(kwargs['derivative_values_log2'], 0),
               **kwargs))


def growth_48h(curve_smooth_growth_data, index48h, *args, **kwargs):
    if index48h < 0 or index48h >= curve_smooth_growth_data.size:
        _logger.warning("Faulty index {0} for 48h size (max {1})".format(index48h, curve_smooth_growth_data.size - 1))
        return np.nan
    return curve_smooth_growth_data[index48h]


def get_chapman_richards_4parameter_extended_curve(x_data, b0, b1, b2, b3, d):
    """Returns a Chapman-Richards 4 parameter log2_curve extended with a
    Y-axis offset d parameter.

    ''Note: The parameters b0, b1, b2 and b3 have been transposed so
    that they stay within the allowed bounds of the model

    Args:

        x_data (np.array):   The X-data

        b0 (float): The first parameter. To ensure that it stays within
                    the allowed bounds b0 > 0, the input b0 is
                    transposed using ``np.power(np.e, b0)``.

        b1 (float): The second parameter. The bounds are
                    1 - b3 < b1 < 1 and thus it is scaled as follows::

                        ``np.power(np.e, b1) / (np.power(np.e, b1) + 1) *
                        b3 + (1 - b3)``

                    Where ``b3`` refers to the transformed version.


        b2 (float): The third parameter, has same bounds and scaling as
                    the first

        b3 (float): The fourth parameter, has bounds 0 < b3 < 1, thus
                    scaling is done with::

                        ``np.power(np.e, b3) / (np.power(np.e, b3) + 1)``

        d (float):  Any real number, used as the offset of the log2_curve,
                    no transformation applied.

    Returns:

        np.array.       An array of matching size as X with the
                        Chapman-Ritchards extended log2_curve for the
                        parameter set.

    """

    # Ensuring parameters stay within the allowed bounds
    b0 = np.power(np.e, b0)
    b2 = np.power(np.e, b2)
    v = np.power(np.e, b3)
    b3 = v / (v + 1.0)
    v = np.power(np.e, b1)
    b1 = v / (v + 1.0) * b3 + (1 - b3)

    return d + b0 * np.power(1.0 - b1 * np.exp(-b2 * x_data), 1.0 / (1.0 - b3))


def get_fit_r_square(x_data, y_data, p0=np.array([1.64, -0.1, -2.46, 0.1, 15.18], dtype=np.float)):
    """x_data and y_data must be 1D, y_data must be log2"""

    finite_y = np.isfinite(y_data)
    x_data = x_data[finite_y]
    y_data = y_data[finite_y]

    try:
        p = leastsq(get_chapman_richards_residuals, p0, args=(x_data, y_data))[0]
    except TypeError:
        return np.inf, p0

    y_hat_vector = get_chapman_richards_4parameter_extended_curve(x_data, *p)

    if y_data.any():
        return (1.0 - np.square(y_hat_vector - y_data).sum() /
                np.square(y_hat_vector - y_data.mean()).sum()), p
    else:
        return np.nan, p


def get_chapman_richards_residuals(cr_params, x_data, y_data):

    return y_data - get_chapman_richards_4parameter_extended_curve(x_data, *cr_params)


def generation_time(derivative_values_log2, index, **kwargs):
    if index < 0:
        _logger.warning("No GT because no finite slopes in data")
        return np.nan
    elif index >= derivative_values_log2.size:
        _logger.warning("Faulty index {0} for GT (max {1})".format(index, derivative_values_log2.size - 1))
        return np.nan
    return 1.0 / derivative_values_log2[index]


def generation_time_error(derivative_errors, index, **kwargs):
    if index < 0:
        _logger.warning("No GT Error because no finite slopes in data")
        return np.nan
    elif index >= derivative_errors.size:
        _logger.warning("Faulty index {0} for GT error (max {1})".format(index, derivative_errors.size - 1))
        return np.nan

    return derivative_errors[index]


def generation_time_when(flat_times, index, linregress_extent, **kwargs):
    pos = index + linregress_extent
    if pos < 0:
        _logger.warning("No GT When because no finite slopes in data")
        return np.nan
    elif pos >= flat_times.size:
        _logger.warning("Faulty index {0} for GT when (max {1})".format(pos, flat_times.size - 1))
        return np.nan

    return flat_times[pos]


def population_size_at_generation_time(curve_smooth_growth_data, index, linregress_extent, **kwargs):
    pos = index + linregress_extent
    if pos < 0:
        _logger.warning("No GT Pop Size because no finite slopes in data")
        return np.nan

    return np.ma.median(
        curve_smooth_growth_data[
            max(0, pos - linregress_extent):
            min(pos + linregress_extent + 1, curve_smooth_growth_data.size)])


def growth_lag(index, flat_times, derivative_values_log2, **kwargs):

    if index < 0:
        return np.nan

    growth_delta = np.log2(population_size_at_generation_time(index=index, **kwargs)) - \
                   np.log2(curve_baseline(**kwargs))

    if growth_delta > 0:

        return np.interp(max(0.0, growth_delta / derivative_values_log2[index]), np.arange(flat_times.size),
                         flat_times)

    return np.nan


def growth_velocity_vector(derivative_values_log2, **kwargs):

    return derivative_values_log2

#
#
# Enum helpers
#

_generation_time_indices = None
_kwargs = None


def _get_generation_time_index(log2_masked_derivative_data, rank):
    finites = log2_masked_derivative_data.size - log2_masked_derivative_data.mask.sum()
    if finites > np.abs(rank):
        return log2_masked_derivative_data.argsort()[:finites][-(rank + 1)]
    return -1


class Phenotypes(Enum):
    """Growth phenotypes from direct study of the growth data.

    They are done on smooth data.

    Each attribute can be called to perform that calculation given that sufficient data is supplied.

    Attributes:
        Phenotypes.InitialValue:
            The first measurement
        Phenotypes.ExperimentFirstTwoAverage:
            Average of first two measurements
        Phenotypes.ExperimentBaseLine:
            Part of `GrowthLag` and `GrowthYield` calculations,
            the average value of first three measurements.
        Phenotypes.ExperimentLowPoint:
            The minimum of first three measurements.
        Phenotypes.ExperimentEndAverage:
            The average of the last three measurements.
            Part of `GrowthYield` calculations.

        Phenotypes.ExperimentGrowthYield:
            The yield of the experiment, difference between beginning
            and end in population size.
        Phenotypes.GrowthLag:
            The duration of how long it took before growth started.
            Intercept between `InitialValue` and the tangent at
            `Phenotypes.GenerationTimeWhen` on log2-scale.

        Phenotypes.GenerationTime48h:
            The generation time at 48h after experiment start
        Phenotypes.ColonySize48h:
            The size of the population at 48h after experiment start

        Phenotypes.GenerationTime:
            The minimum time of population doubling
        Phenotypes.GenerationTimeStErrOfEstimate:
            The standard error of the estimate of the linear
            regression that is the basis of `GenerationTime`.
        Phenotypes.GenerationTimeWhen:
            When during the experiment `GenerationTime` occurred.
        Phenotypes.GenerationTimePopulationSize:
            The population size when `GenerationTime` occurred.

        Phenotypes.GenerationTime2:
            The second shortest time of population doubling
        Phenotypes.GenerationTime2StErrOfEstimate:
            The standard error of the estimate of the linear
            regression that is the basis of
            `Phenotypes.GenerationTime2`.
        Phenotypes.GenerationTime2When:
            When during the experiment `GenerationTime2` occurred.

        Phenotypes.ChapmanRichardsFit:
            How well the Chapman-Richards growth model fit the data
        Phenotypes.ChapmanRichardsParam1:
            The first parameter of the model
        Phenotypes.ChapmanRichardsParam2:
            The second parameter of the model
        Phenotypes.ChapmanRichardsParam3:
            The third parameter of the model
        Phenotypes.ChapmanRichardsParam4:
            The fourth parameter of the model
        Phenotypes.ChapmanRichardsParamXtra:
            The additional parameter (population size at start of
            experiment).

        Phenotypes.ExperimentPopulationDoublings:
            The `GrowthYield` recalculated as population size
            doublings.
            _NOTE_: This does not directly imply number of cell
            divisions or generations (death exisits!).

        Phenotypes.ResidualGrowth:
            The amount of growth that happens after time of maximum
            growth (`Phenotypes.GenerationTimeWhen`).
        Phenotypes.ResidualGrowthAsPopulationDoublings:
            The number of times the population doubles after time
            of maximum growth (`Phenotypes.GenerationTimeWhen`).
        Phenotypes.GrowthVelocityVector:
            The derivative of the growth data.

    """
    InitialValue = 12
    """:type Phenotypes"""
    ExperimentFirstTwoAverage = 13
    """:type Phenotypes"""
    ExperimentBaseLine = 14
    """:type Phenotypes"""
    ExperimentLowPoint = 15
    """:type Phenotypes"""
    ExperimentLowPointWhen = 23
    """:type Phenotypes"""
    ExperimentEndAverage = 16
    """:type Phenotypes"""

    ExperimentGrowthYield = 17
    """:type Phenotypes"""
    GrowthLag = 18
    """:type Phenotypes"""

    GenerationTime48h = 19
    """:type Phenotypes"""
    ColonySize48h = 20
    """:type Phenotypes"""

    GenerationTime = 0
    """:type Phenotypes"""
    GenerationTimeStErrOfEstimate = 1
    """:type Phenotypes"""
    GenerationTimeWhen = 2
    """:type Phenotypes"""
    GenerationTimePopulationSize = 21
    """:type Phenotypes"""

    GenerationTime2 = 3
    """:type Phenotypes"""
    GenerationTime2StErrOfEstimate = 4
    """:type Phenotypes"""
    GenerationTime2When = 5
    """:type Phenotypes"""

    ChapmanRichardsFit = 6
    """:type Phenotypes"""
    ChapmanRichardsParam1 = 7
    """:type Phenotypes"""
    ChapmanRichardsParam2 = 8
    """:type Phenotypes"""
    ChapmanRichardsParam3 = 9
    """:type Phenotypes"""
    ChapmanRichardsParam4 = 10
    """:type Phenotypes"""
    ChapmanRichardsParamXtra = 11
    """:type Phenotypes"""

    ExperimentPopulationDoublings = 22
    """:type Phenotypes"""

    ResidualGrowth = 24
    """:type Phenotypes"""
    ResidualGrowthAsPopulationDoublings = 25
    """:type Phenotypes"""

    Monotonicity = 26
    """:type Phenotypes"""

    GrowthVelocityVector = 1000
    """:type Phenotypes"""

    def __call__(self, **kwargs):

        if self is Phenotypes.InitialValue:
            return initial_value(**kwargs)

        elif self is Phenotypes.ExperimentBaseLine:
            return curve_baseline(**kwargs)

        elif self is Phenotypes.ExperimentEndAverage:
            return curve_end_average(**kwargs)

        elif self is Phenotypes.ExperimentFirstTwoAverage:
            return curve_first_two_average(**kwargs)

        elif self is Phenotypes.ColonySize48h:
            return growth_48h(**kwargs)

        elif self is Phenotypes.GenerationTime48h:
            return generation_time(index=kwargs['index48h'], **kwargs)

        elif self is Phenotypes.ExperimentGrowthYield:
            return growth_yield(**kwargs)

        elif self is Phenotypes.ExperimentPopulationDoublings:
            return growth_curve_doublings(**kwargs)

        elif self is Phenotypes.ResidualGrowth:
            return residual_growth(**kwargs)

        elif self is Phenotypes.ResidualGrowthAsPopulationDoublings:
            return residual_growth_as_population_doublings(**kwargs)

        elif self is Phenotypes.ExperimentLowPoint:
            return curve_low_point(**kwargs)

        elif self is Phenotypes.ExperimentLowPointWhen:
            return curve_low_point_time(**kwargs)

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
            return generation_time(index=_get_generation_time_index(kwargs['derivative_values_log2'], 0), **kwargs)

        elif self is Phenotypes.GenerationTime2:
            return generation_time(index=_get_generation_time_index(kwargs['derivative_values_log2'], 1), **kwargs)

        elif self is Phenotypes.GenerationTimeWhen:
            return generation_time_when(index=_get_generation_time_index(kwargs['derivative_values_log2'], 0), **kwargs)

        elif self is Phenotypes.GenerationTime2When:
            return generation_time_when(index=_get_generation_time_index(kwargs['derivative_values_log2'], 1), **kwargs)

        elif self is Phenotypes.GenerationTimeStErrOfEstimate:
            return generation_time_error(index=_get_generation_time_index(kwargs['derivative_values_log2'], 0), **kwargs)

        elif self is Phenotypes.GenerationTime2StErrOfEstimate:
            return generation_time_error(index=_get_generation_time_index(kwargs['derivative_values_log2'], 1), **kwargs)

        elif self is Phenotypes.GenerationTimePopulationSize:
            return population_size_at_generation_time(index=_get_generation_time_index(kwargs['derivative_values_log2'], 0), **kwargs)

        elif self is Phenotypes.GrowthLag:
            return growth_lag(index=_get_generation_time_index(kwargs['derivative_values_log2'], 0), **kwargs)

        elif self is Phenotypes.GrowthVelocityVector:
            return growth_velocity_vector(**kwargs)

        elif self is Phenotypes.Monotonicity:
            return curve_monotonicity(**kwargs)
