#
#   DEPENDENCIES
#

from enum import Enum
import numpy as np
from types import StringTypes
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel, laplace, convolve, generic_filter, median_filter
from scipy.stats import pearsonr

#
#   INTERNAL DEPENDENCIES
#

from scanomatic.data_processing.data_bridge import Data_Bridge
from scanomatic.generics.maths import mid50_mean
from scanomatic.generics.phenotype_filter import FilterArray, Filter

#
#   STATIC GLOBALS
#


class Offsets(Enum):
    """Subplate offsets with orientation
    (`scanomatic.data_processing.phenotyper.Phenotyper` data)

    Indices enumerate from 0.

    Attributes:
        Offsets.LowerRight: Outer index even, inner index even
        Offsets.LowerLeft: Outer index even, inner index odd
        Offsets.UpperLeft: Outer index odd, inner index odd
        Offsets.UpperRight: Outer index odd, inner index even
    """
    LowerRight = 0
    LowerLeft = 1
    UpperLeft = 2
    UpperRight = 3

    def __call__(self):

        return np.array([[self is Offsets.UpperLeft, self is Offsets.UpperRight],
                         [self is Offsets.LowerLeft, self is Offsets.LowerRight]], dtype=np.bool)


def infer_offset(arr):

    for offset in Offsets:

        if (offset() == arr).all():
            return offset

    raise ValueError("Supplied pattern {0} not a known offset".format(arr.tolist()))

#
#   METHODS: Normalisation methods
#


def get_downsampled_plates(data, subsampling="BR"):
    """

        The subsampling is either supplied as a generic position for all plates
        or as a list of individual positions for each plate of the dataBridge
        or object following the same interface (e.g. numpy array of plates).

        The subsampling will attempt to extract one of the four smaller sized
        plates that made up the current plate. E.g. one of the four 384 plates
        that makes up a 1536 as the 384 looked before being pinned over.

        The returned array will work on the same memory as the original so
        any changes will affect the original data.

        The subSampling should be one of the following four expressions:

            TL:     Top left
            TR:     Top right
            BL:     Bottom left
            BR:     Bottom right

        Args:
            data:
                The data
            subsampling:
                The subsampling used
    """

    # Generic -> Per plate
    if isinstance(subsampling, StringTypes):
        subsampling = [subsampling for _ in range(data.shape[0])]

        # Lookup to translate subsamplingexpressions to coordinates
        sub_sample_lookup = {'TL': Offsets.UpperLeft, 'TR': Offsets.UpperRight,
                             'BL': Offsets.LowerLeft, 'BR': Offsets.LowerRight}

        # Name to offset
        subsampling = [sub_sample_lookup[s]() for s in subsampling]

    # Create a new container for the plates. It is important that this remains
    # a list and is not converted into an array if both returned members of A
    # and original plate values should operate on the same memory

    # Number of dimensions to subsample, take first two
    # subsample_first_dim = 2

    out = []
    for i, plate in enumerate(data):
        offset = subsampling[i]

        if offset.astype(bool).sum() != 1:
            raise ValueError(
                "Only exactly 1 reference position offset per plate allowed. "
                "You had {0}".format(offset))
        elif offset.shape != (2, 2):
            raise ValueError(
                "Only subsampling by 2x2 arrays allowed. You had {0}".format(
                    offset.shape))

        d1, d2 = np.where(offset)

        out.append(plate[d1[0]::2, d2[0]::2])

    return out


def get_control_position_filtered_arrays(data, offsets=None, fill_value=np.nan):

    """Support method that returns array in the shape corresponding
    to the data in the DataBridge such that only the values reported
    in the control positions are maintained (without affecting the contents
    of the Databridge).

    Args:
        data:
            The data
        offsets:
            Control position offsets
        fill_value:
            Value to fill non control positions with
    """

    if isinstance(data, Data_Bridge):
        data = data.get_as_array()
    else:
        for i, d in enumerate(data):
            if isinstance(d, FilterArray):
                data[i] = d.masked(Filter.UndecidedProblem, Filter.BadData, Filter.Empty).filled()

    n_plates = len(data)
    out = []

    if offsets is None:
        offsets = [Offsets.LowerRight() for _ in range(n_plates)]

    for id_plate in xrange(n_plates):

        new_plate = data[id_plate].copy()
        out.append(new_plate)
        plate_offset = offsets[id_plate]
        filt = np.tile(plate_offset, [a / b for a, b in zip(new_plate.shape, plate_offset.shape)])
        new_plate[filt == False] = fill_value

    return np.array(out)


def _get_positions(data, offsets):

    out = []

    for id_plate, plate in enumerate(data):

        offset = offsets[id_plate]
        filt = np.tile(offset, [a / b for a, b in zip(plate.shape, offset.shape)])
        filt = filt.reshape(filt.shape + tuple(1 for _ in range(plate.ndim - filt.ndim)))
        out.append(np.where(filt & np.isfinite(plate)))

    return out


def get_control_position_coordinates(data, offsets=None):
    """Returns list of tuples that emulates the results of running np.where

    Args:
        data:
            The data
        offsets:
            Control position offsets
    """

    n_plates = len(data)

    if offsets is None:
        offsets = [Offsets.LowerRight() for _ in range(n_plates)]

    return _get_positions(data, offsets)


def get_experiment_positions_coordinates(data, offsets=None):

    if offsets is None:
        offsets = [Offsets.LowerRight() for _ in range(len(data))]

    experiment_positions_offsets = [k == np.False_ for k in offsets]

    return _get_positions(data, experiment_positions_offsets)


def get_coordinate_filtered(data, coordinates, measure=1, require_finite=True, require_correlated=False):

    if isinstance(data, Data_Bridge):
        data = data.get_as_array()

    filtered = []
    for i in range(len(data)):

        p = data[i][..., measure]
        filtered_plate = p[coordinates[i]]

        if require_finite and not require_correlated:
            filtered_plate = filtered_plate[np.isfinite(filtered_plate)]

        filtered.append(filtered_plate)

    filtered = np.array(filtered)

    if require_correlated:

        filtered = filtered[:, np.isfinite(filtered).all(axis=0)]

    return filtered


def get_center_transformed_control_positions(control_pos_coordinates, data):

    """Remaps coordinates so they are relative to the plates' center

    Args:
        control_pos_coordinates:
            Positions of the controls

        data:
            The data
    """

    center_transformed = []

    if isinstance(data, Data_Bridge):
        data = data.get_as_array()

    for id_plate, plate in enumerate(control_pos_coordinates):

        center = data[id_plate].shape[:2] / 2.0
        center_transformed.append((plate[0] - center[0],  plate[1] - center[1]))

    return center_transformed


def get_control_positions_average(control_pos_data_array,
                                  overwrite_experiment_values=np.nan,
                                  average_method=mid50_mean):
    """Returns the average per measure of each measurement type for
    the control positions. Default is to return the mean of the
    the mid 50 values.

    Args:
        control_pos_data_array:
            The data
        overwrite_experiment_values:
            Which data not to include
        average_method:
            The averaging method used
    """

    plate_control_averages = []

    for plate in control_pos_data_array:

        measurement_vector = []
        plate_control_averages.append(measurement_vector)

        for id_measurement in xrange(plate.shape[2]):

            if overwrite_experiment_values in (np.nan, np.inf):

                if np.isnan(overwrite_experiment_values):

                    valid_data_test = np.isnan

                else:

                    valid_data_test = np.isinf

                measurement_vector.append(
                    average_method(plate[..., id_measurement][
                        valid_data_test(plate[..., id_measurement]) == np.False_]))

            else:

                measurement_vector.append(
                    average_method(plate[..., id_measurement][
                                      plate[..., id_measurement] != overwrite_experiment_values]))

    return np.array(plate_control_averages)


def get_normalisation_surface(control_positions_filtered_data, control_position_coordinates=None,
                              norm_sequence=('cubic', 'linear', 'nearest'), use_accumulated=False, fill_value=np.nan,
                              offsets=None, apply_median_smoothing_kernel=None, apply_gaussian_smoothing_sigma=None):
    """Constructs normalisation surface using iterative runs of
    scipy.interpolate's gridddata based on sequence of supplied
    method preferences.

    Args:
        control_positions_filtered_data:
            An array with only control position values intact.
            All other values should be missingDataValue or they won't be
            calculated.

        control_position_coordinates:
            Optional argument to supply already constructed
            per plate control positions vector. If not supplied
            it is constructed using controlPositionKernel

        norm_sequence (str):
            ('cubic', 'linear', 'nearest')
            The griddata method order to be invoked.

        use_accumulated (False):
            If later stage methods should use information obtained in
            earlier stages or only work on original control positions.

        fill_value (np.nan):
            The value to be used to indicate that normalisation value
            for a position is not known

        offsets (None):
            Argument passed on when constructing the
            controlPositionsCoordinates if it is not supplied.

        apply_median_smoothing_kernel (int):
            Optional argument to apply a median smoothing of certain size to the surface
            after interpolation. Omitted if `None`.

        apply_gaussian_smoothing_sigma (float):
            Optional argument to apply a gaussian smoothing with sigma.
            Omitted if `None`.

    """

    def get_stripped_invalid_points(selector):

        return tuple(map(np.array, zip(*((x, y) for is_selected, x, y in zip(selector, *anchor_points)
                                         if is_selected))))

    out = []
    n_plates = len(control_positions_filtered_data)

    if control_position_coordinates is None:
        control_position_coordinates = get_control_position_coordinates(control_positions_filtered_data, offsets)

    if np.isnan(fill_value):

        missing_data_test = np.isnan

    else:

        missing_data_test = np.isinf

    for id_plate in xrange(n_plates):

        if control_positions_filtered_data[id_plate] is None:
            out.append(None)
            continue

        anchor_points = control_position_coordinates[id_plate]
        plate = control_positions_filtered_data[id_plate].copy()
        if plate.ndim == 2:
            anchor_points = anchor_points[:2]
        out.append(plate)
        grid_x, grid_y = np.mgrid[0:plate.shape[0], 0:plate.shape[1]]

        if plate.ndim == 3:
            for id_measurement in xrange(plate.shape[2]):

                for method in norm_sequence:

                    if use_accumulated:
                        anchor_points = np.where(missing_data_test(plate[..., id_measurement]) == np.False_)
                    anchor_values = plate[anchor_points]

                    finite_values = np.isfinite(anchor_values)
                    if finite_values.sum() > 0:

                        if (finite_values == np.False_).any():

                            anchor_points = tuple(p[finite_values] for p in anchor_points)
                            anchor_values = anchor_values[finite_values]

                        splined_plate = griddata(tuple(anchor_points[:2]), anchor_values, (grid_x, grid_y), method=method,
                                                 fill_value=fill_value)
                        missing_data = np.where(missing_data_test(plate[..., id_measurement]))
                        plate[..., id_measurement][missing_data] = splined_plate[missing_data]

                        if not missing_data_test(plate[..., id_measurement]).any():
                            break
        else:
            for method in norm_sequence:

                if use_accumulated:
                    anchor_points = np.where(missing_data_test(plate) == np.False_)

                anchor_points = get_stripped_invalid_points(np.isfinite(plate[anchor_points]))
                anchor_values = plate[anchor_points]

                if anchor_values.size == 0 or np.isfinite(plate).all() or np.isfinite(anchor_values).sum() < 4:
                    break
                else:

                    splined_plate = griddata(anchor_points, anchor_values, (grid_x, grid_y), method=method,
                                             fill_value=fill_value)
                    missing_data = np.where(missing_data_test(plate))
                    plate[missing_data] = splined_plate[missing_data]

                    if not missing_data_test(plate).any():
                        break

    out = np.array(out)

    if apply_median_smoothing_kernel is not None:
        for id_measurement in xrange(out[0].shape[2]):
            apply_median_smoothing(
                out,
                filter_shape=apply_median_smoothing_kernel,
                measure=id_measurement)

    if apply_gaussian_smoothing_sigma is not None:
        for id_measurement in xrange(out[0].shape[2]):
            apply_gauss_smoothing(
                out,
                sigma=apply_gaussian_smoothing_sigma,
                measure=id_measurement)

    return out

#
#   METHODS: Apply functions
#
#   Apply functions update the dataArray/Bridge values!
#


def apply_outlier_filter(data, median_filter_size=(3, 3), measure=None, k=2.0, p=10, max_iterations=10):
    """Checks all positions in each array and filters those outside
    set boundries based upon their peak/valey properties using
    laplace and normal distribution assumptions.

    Args:
        data (numpy.array):    Array of platewise values

        median_filter_size (tuple):    Used in median filter that is nan-safe as
                                first smoothing step before testing outliers.
                                If set to None, step is skipped

        measure (int):  The measure to be outlier filtered

        k (float) : Distance in sigmas for setting nan-threshold

        p (int) :   Estimate number of positions to be qualified as outliers

        max_iterations (int) :   Maximum number of iterations filter may be
                                applied
    """

    def nan_filler(item):
        if np.isnan(item[kernel_center]):

            return np.median(item[np.isfinite(item)])

        else:

            return item[kernel_center]

    if median_filter_size is not None:

        kernel_center = (np.prod(median_filter_size) - 1) / 2

        assert np.array([v % 2 == 1 for v in median_filter_size]).all(), "nanFillSize can only have odd values"

    laplace_kernel = np.array([
        [0.5, 1, 0.5],
        [1, -6, 1],
        [0.5, 1, 0.5]], dtype=data[0].dtype)

    for plate in data:

        old_nans = -1
        new_nans = 0
        iterations = 0

        while new_nans != old_nans and iterations < max_iterations:

            old_nans = new_nans
            iterations += 1

            if measure is None:
                plate_copy = plate.copy()
            else:
                plate_copy = plate[..., measure].copy()

            if median_filter_size is not None:

                # Apply median filter to fill nans
                plate_copy = generic_filter(plate_copy, nan_filler, size=median_filter_size, mode="nearest")

            # Apply laplace
            plate_copy = convolve(plate_copy, laplace_kernel, mode="nearest")

            # Make normalness analysis to find lower and upper threshold
            # Rang based to z-score, compare to threshold adjusted by expected
            # fraction of removed positions
            plate_copy_ravel = np.ma.masked_invalid(plate_copy.ravel())
            if measure is None:
                plate_ravel = plate.ravel()
            else:
                plate_ravel = plate[..., measure].ravel()
            sigma = plate_copy_ravel.std()
            mu = plate_copy_ravel.mean()
            z_scores = np.abs(plate_copy_ravel.data - mu)

            for pos in np.argsort(z_scores)[::-1]:
                if np.isnan(plate_ravel[pos]) or \
                        z_scores[pos] > k * sigma / np.exp(-(np.isfinite(plate_ravel).sum() /
                                                             float(plate_ravel.size)) ** p):

                    if measure is None:
                        plate[pos / plate.shape[1], pos % plate.shape[1]] = np.nan
                    else:
                        plate[pos / plate.shape[1], pos % plate.shape[1], measure] = np.nan

                else:

                    break

            if measure is None:
                new_nans = np.isnan(plate).sum()
            else:
                new_nans = np.isnan(plate[..., measure]).sum()


def apply_log2_transform(data, measures=None):
    """Log2 Transformation of dataArray values.

    If required, a filter for which measures to be log2-transformed as
    either an array or tuple of measure indices. If left None, all measures
    will be logged

    Args:
        data:       Data to be transformed
        measures:   If only a specific measure should be transformed
    """

    if measures is None:
        measures = np.arange(data[0].shape[-1])

    for id_plate in range(len(data)):
        data[id_plate][..., measures] = np.log2(
            data[id_plate][..., measures])


def apply_sobel_filert(data, measure=1, threshold=1, **kwargs):
    """Applies a Sobel filter to the arrays and then compares this to a
    threshold setting all positions greater than said absolute threshold to NaN.

    Args:
        data:       The data to be filtered
        measure:    The measurement to evaluate
        threshold:  The maximum absolute value allowed

    Further arguments of `scipy.ndimage.sobel` can be supplied
    """

    if 'mode' not in kwargs:
        kwargs['mode'] = 'nearest'

    for id_plate in range(len(data)):

        filt = (np.sqrt(sobel(
            data[id_plate][..., measure], axis=0, **kwargs) ** 2 +
                        sobel(data[id_plate][..., measure], axis=1, **kwargs) ** 2) > threshold)

        data[id_plate][..., measure][filt] = np.nan


def apply_laplace_filter(data, measure=1, threshold=1, **kwargs):
    """Applies a Laplace filter to the arrays and then compares the absolute
    values of those to a threshold, discarding those exceeding it.

    Args:
        data:       The data to be filtered
        measure:    The measurement to evaluate
        threshold:  The maximum absolute value allowed

    Further arguments of `scipy.ndimage.laplace` can be supplied
    """
    if 'mode' not in kwargs:
        kwargs['mode'] = 'nearest'

    for id_plate in range(len(data)):

        filt = np.abs(laplace(data[id_plate][..., measure], **kwargs)) > threshold

        data[id_plate][..., measure][filt] = np.nan


def apply_gauss_smoothing(data, measure=1, sigma=3.5, **kwargs):
    """Applies a Gaussian Smoothing filter to the values of a plate (or norm
    surface).

    Note that this will behave badly if there are NaNs on the plate.

    Args:
        data:       The data to be smoothed
        measure:    The measurement ot evaluate
        sigma:      The size of the gaussian kernel
    """

    if 'mode' not in kwargs:
        kwargs['mode'] = 'nearest'

    for id_plate in range(len(data)):

        data[id_plate][..., measure] = gaussian_filter(
            data[id_plate][..., measure], sigma=sigma, **kwargs)


def apply_median_smoothing(data, measure=1, filter_shape=(3, 3), **kwargs):

    if 'mode' not in kwargs:
        kwargs['mode'] = 'nearest'

    for id_plate, plate in enumerate(data):

        data[id_plate][..., measure] = median_filter(plate[..., measure], size=filter_shape, **kwargs)


def apply_sigma_filter(data, sigma=3):
    """Applies a per plate global sigma filter such that those values
    exceeding the absolute sigma distance to the mean are discarded.

    Args:
        data: Data to be filtered
        sigma: Threshold distance from mean
    """
    for id_plate in range(len(data)):

        for measure in range(data[id_plate].shape[-1]):

            values = data[id_plate][..., measure]
            finite_values = values[np.isfinite(values)]
            mean = finite_values.mean()
            std = finite_values.std()
            values[np.logical_or(values < mean - sigma * std,
                                 values > mean + sigma * std)] = np.nan


#
#   FITTING METHODS: For normalisation with GT and inital value
#


def initial_plate_transform(ipv, flex_scaling_vector, magnitude_scaling_vector):

    return magnitude_scaling_vector * (flex_scaling_vector * (ipv - ipv[np.isfinite(ipv)].mean()))


def ipv_residue(scaling_params, ipv, gt):

    ip_flex = scaling_params[: scaling_params.size / 2]
    ip_scale = scaling_params[scaling_params.size / 2:]
    ret = np.array([gt[id_p] - initial_plate_transform(
        ipv, ip_flex[id_p], ip_scale[id_p]) for id_p in xrange(gt.shape[0])
                     if gt[id_p].size > 0], dtype=np.float)

    return np.hstack([p[np.isfinite(p)].ravel() for p in ret])

#
#   METHODS: Normalisation method
#


def norm_by_log2_diff(plate, surface, **kwargs):
    return np.log2(plate) - np.log2(surface)


def norm_by_diff(plate, surface, **kwargs):
    return plate - surface


def norm_by_signal_to_noise(plate, surface, std, **kwargs):
    return (plate - surface) / std


def norm_by_log2_diff_corr_scaled(plate, surface, **kwargs):
    plate = np.log2(plate)
    surface = np.log2(surface)
    filt = np.isfinite(plate) & np.isfinite(surface)
    return (plate - surface) * (pearsonr(plate[filt], surface[filt])[0] + 1) * 0.5


def get_normalized_data(data, offsets=None, method=norm_by_log2_diff):

    if data is None:
        return None

    surface = get_control_position_filtered_arrays(data, offsets=offsets)

    pre_surface = get_downsampled_plates(surface, offsets)
    apply_outlier_filter(pre_surface, measure=None)

    std = [None] * len(data)

    if method == norm_by_signal_to_noise:
        std = [plate[np.isfinite(plate)].std() if plate is not None else None for plate in pre_surface]

    try:
        surface = get_normalisation_surface(surface, offsets=offsets)
    except ValueError:
        print offsets
        print data
        raise

    return normalisation(data, surface, method=method, std=std)


def get_reference_positions(data, offsets, outlier_filter=True):

    surface = get_control_position_filtered_arrays(data, offsets=offsets)
    pre_surface = get_downsampled_plates(surface, offsets)
    if outlier_filter:
        apply_outlier_filter(pre_surface, measure=None)
    return pre_surface


def normalisation(data, norm_surface, method=norm_by_log2_diff, std=(None,)):

    normed_data = []
    if isinstance(data, Data_Bridge):
        data = data.get_as_array()

    for id_plate, (plate, surf, plate_std) in enumerate(zip(data, norm_surface, std)):

        if plate is None or surf is None:
            normed_data.append(None)

        else:
            normed_data.append(method(plate, surf, std=plate_std))

    return np.array(normed_data)
