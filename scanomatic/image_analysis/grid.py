import numpy as np

try:
    from skimage import filters as ski_filter
except ImportError:
    from skimage import filter as ski_filter

from scipy import ndimage

#
# FUNCTIONS
#


def get_adaptive_threshold(im, threshold_filter=None, segments=60,
                           sigma=None, *args, **kwargs):
    """Gives a 2D surface of threshold based on smoothed local measures"""

    if threshold_filter is None:
        threshold_filter = ski_filter.threshold_otsu
    if sigma is None:
        sigma = np.sqrt(im.size) / 5

    if segments is None or segments == 5:
        # TODO: Hack solution, make nice
        segmented_image = np.zeros(im.shape)
        segmented_image[im.shape[0] / 4, im.shape[1] / 4] = 1
        segmented_image[im.shape[0] / 4, im.shape[1] * 3 / 4] = 1
        segmented_image[im.shape[0] * 3 / 4, im.shape[1] / 4] = 1
        segmented_image[im.shape[0] * 3 / 4, im.shape[1] * 3 / 4] = 1
        segmented_image[im.shape[0] / 2, im.shape[1] / 2] = 1
    else:
        p = 1 - np.float(segments) / im.size
        segmented_image = (np.random.random(im.shape) > p).astype(np.uint8)

    labled, labels = _get_sectioned_image(segmented_image)

    for l in range(1, labels + 1):

        l_filter = labled == l
        if l_filter.sum() > 1:

            i_slice = im[l_filter]
            if i_slice.std() != 0:

                segmented_image[ndimage.binary_dilation(l_filter, iterations=4)] = \
                    threshold_filter(i_slice, *args, **kwargs)

            else:

                segmented_image[ndimage.binary_dilation(l_filter, iterations=4)] = \
                    i_slice.mean()

    return ndimage.gaussian_filter(segmented_image, sigma=sigma)


def _get_sectioned_image(im):
    """Sections image in proximity regions for points of interests"""

    distance_image = ndimage.distance_transform_edt(im == 0)
    kernel = np.array([[-1, 2, -1]])
    d2 = ndimage.convolve(distance_image, kernel) + ndimage.convolve(distance_image, kernel.T)
    d2 = ndimage.binary_dilation(d2 > d2.mean(), border_value=1) == 0
    labeled, labels = ndimage.label(d2)
    return labeled, labels


def get_denoise_segments(im, **kwargs):
    """Filters out small segments"""
    erode_im = ndimage.binary_erosion(im, **kwargs)
    reconstruct_im = ndimage.binary_propagation(erode_im, mask=im)
    tmp = np.logical_not(reconstruct_im)
    erode_tmp = ndimage.binary_erosion(tmp, **kwargs)
    reconstruct_final = np.logical_not(ndimage.binary_propagation(
        erode_tmp, mask=tmp))

    return reconstruct_final


def get_segments_by_size(im, min_size, max_size=-1, inplace=True):
    """Filters segments by allowed size range"""

    if inplace:
        out = im
    else:
        out = im.copy()

    if max_size == -1:
        max_size = im.size

    labled_im, labels = ndimage.label(im)
    sizes = ndimage.sum(im, labled_im, range(labels + 1))

    mask_sizes = np.logical_or(sizes < min_size, sizes > max_size)
    remove_pixels = mask_sizes[labled_im]

    out[remove_pixels] = False

    if not inplace:
        return out


def get_segments_by_shape(im, max_shape, check_roundness=True,
                          inplace=True):
    if inplace:
        out = im
    else:
        out = im.copy()

    labeled_im, labels = ndimage.label(im)
    segments = ndimage.find_objects(labeled_im)

    bound_d1, bound_d2 = max_shape

    for i, segment in enumerate(segments):

        s_d1, s_d2 = segment
        if (abs(s_d1.stop - s_d1.start) > bound_d1 or
                abs(s_d2.stop - s_d2.start) > bound_d2):

            out[segment][labeled_im[segment] == i + 1] = False

        elif check_roundness:

            feature_slice = im[segment]
            blob = feature_slice[labeled_im[segment] == i + 1]

            if is_almost_square(feature_slice) or is_almost_round(feature_slice, blob):

                out[segment][labeled_im[segment] == i + 1] = False

    return out

_DEVIATION_FROM_SQUARE_TOLERANCE = 0.25


def is_almost_square(feature_slice):
    global _DEVIATION_FROM_SQUARE_TOLERANCE
    return abs(1 - float(feature_slice.shape[0]) / feature_slice.shape[1]) > _DEVIATION_FROM_SQUARE_TOLERANCE

_DEVIATION_FROM_CIRCLE_TOLERANCE = 0.1
_INSET_CIRCLE_IN_SQUARE_FACTOR = np.pi / 4.0


def is_almost_round(feature_slice, blob):
    global _DEVIATION_FROM_CIRCLE_TOLERANCE
    global _INSET_CIRCLE_IN_SQUARE_FACTOR
    return abs(1 - feature_slice.size * _INSET_CIRCLE_IN_SQUARE_FACTOR / blob.sum()) > _DEVIATION_FROM_CIRCLE_TOLERANCE


def get_grid_parameters(x_data, y_data, grid_shape, spacings=(54, 54)):

    data = (x_data, y_data)
    new_spacings = get_grid_spacings(x_data, y_data, *spacings)
    if None in new_spacings:
        return None, None
    centers = get_centre_candidates(grid_shape, new_spacings)
    votes = get_votes(data, centers)
    weights = get_weights(votes, data, 1.0)
    sigma = np.max((spacings[0], new_spacings[1])) * 0.1 / np.sqrt(2) + 0.5
    heatmap = get_heatmap(data, votes, weights, sigma)

    _center_dim2, _center_dim1 = np.unravel_index(
        heatmap.argmax(), heatmap.shape)

    new_center = (_center_dim1, _center_dim2)

    return new_center, new_spacings


def get_weights(votes, data, width=1.0):
    """
    Get weights for votes. If width > 0, a Gaussian weight is assigend based
    on the distance of the vote to the mean of the data. The width of the
    Gaussian is set from the spread of the data, and scaled by the width
    parameter. By default, width is set to be 1, which is a very weak
    weighting.
    """

    x_data, y_data = data
    x_votes, y_votes = votes

    if width > 0:
        weights = (np.exp(-((x_votes - x_data.mean()) ** 2 / (width * x_data.std()) ** 2 +
                   (y_votes - y_data.mean()) ** 2 / (width * y_data.std()) ** 2)) /
                   (2 * np.pi * x_data.std() * y_data.std() * width ** 2))
    else:
        weights = np.ones(x_votes.shape)

    return weights


# noinspection PyUnresolvedReferences
def get_votes(data, centers):
    """
    Get votes from all data points.
    """

    x_data, y_data = data
    x_centers, y_centers = centers

    x_votes = np.add.outer(x_data, x_centers)
    y_votes = np.add.outer(y_data, y_centers)

    return x_votes.ravel(), y_votes.ravel()


def get_heatmap(data, votes, weights, sigma):
    """
    Get smoothed histogram.

    A good value for sigma is probably  max(dx, dy) * leeway /sqrt(2) + 0.5.
    """

    x_data, y_data = data
    x_votes, y_votes = votes

    vote_slice = np.logical_and(np.logical_and(x_votes >= 0, y_votes >= 0),
                                np.logical_and(x_votes <= x_data.max(), y_votes <= y_data.max()))

    x_votes = x_votes[vote_slice]
    y_votes = y_votes[vote_slice]
    votes_weights = weights[vote_slice]

    heatmap = np.zeros((int(np.ceil(y_data.max()) + 1), int(np.ceil(x_data.max()) + 1)))

    x_votes = np.round(x_votes).astype(np.int)
    y_votes = np.round(y_votes).astype(np.int)

    flat_votes_xy = y_votes * heatmap.shape[1] + x_votes

    unique_votes = np.unique(flat_votes_xy)
    unique_votes.sort()

    def get_between_votes_bins(votes):
        index_bin_offset = 0.5
        return get_appended_vote_bins(votes) + index_bin_offset

    def get_appended_vote_bins(votes):
        first_bin_edge = -1
        return np.hstack(((first_bin_edge,), votes))

    unique_vote_weights, _ = np.histogram(
        flat_votes_xy, bins=get_between_votes_bins(unique_votes),
        weights=votes_weights)

    heatmap.ravel()[unique_votes] = unique_vote_weights

    if sigma > 0:
        heatmap = ndimage.gaussian_filter(heatmap, sigma)

    if heatmap.sum() > 0:
        heatmap /= heatmap.sum()

    return heatmap


def get_centre_candidates(grid_size, spacings):
    """
    Get coordinates of all possible grid centres without offset.
    """

    dx, dy = spacings

    grid_y, grid_x = np.mgrid[
        -(grid_size[1] - 1) * 0.5: (grid_size[1] - 1) * 0.5 + 1,
        -(grid_size[0] - 1) * 0.5: (grid_size[0] - 1) * 0.5 + 1]

    return grid_x.ravel() * dx, grid_y.ravel() * dy


def get_grid_spacings(x_data, y_data, expected_dx, expected_dy, leeway=0.1):

    # TODO: Remove expected values and use fourier transform to get the rough frequency

    # noinspection PyUnresolvedReferences
    def get_delta(data, expected_delta):
        deltas = np.abs(np.subtract.outer(data, data))
        filt = np.logical_and(
            deltas > expected_delta * (1 - leeway),
            deltas < expected_delta * (1 + leeway))

        if filt.any():
            return deltas[filt].mean()
        return None

    return get_delta(x_data, expected_dx), get_delta(y_data, expected_dy)


def replace_ideal_with_observed(ideal_grid, x_data, y_data, max_sq_dist):

    shape = np.array(ideal_grid.shape[1:])
    update_allowed = np.ones(shape, dtype=np.bool)

    rings = shape / 2

    # noinspection PyUnresolvedReferences
    def _look_replace(array_view, filt):

        ideal_xs = array_view[0].ravel()
        ideal_ys = array_view[1].ravel()
        distances = np.subtract.outer(ideal_xs, x_data) ** 2 + np.subtract.outer(ideal_ys, y_data) ** 2
        replace_positions = distances.min(axis=1) <= max_sq_dist
        best_fit_indices = distances.argmin(axis=1)[replace_positions]
        where_pos = np.unravel_index(np.where(replace_positions)[0], filt.shape)
        for i in xrange(where_pos[0].size):
            d1 = where_pos[0][i]
            d2 = where_pos[1][i]
            if filt[d1, d2]:
                section[0, d1, d2] = x_data[best_fit_indices[i]]
                section[1, d1, d2] = y_data[best_fit_indices[i]]

    def _push_ideal(array_view, current_ring):

        if (rings[0] - current_ring - 1) >= 0:
            distance_dim1_lower = ideal_grid[0, rings[0] - current_ring - 1, :].mean() - array_view[0, 0, :].mean()
            distance_dim1_upper = ideal_grid[0, rings[0] + current_ring, :].mean() - array_view[0, -1, :].mean()
            distance_dim1_lower_slice = np.s_[0, : rings[0] - current_ring, :]
            ideal_grid[distance_dim1_lower_slice][update_allowed[distance_dim1_lower_slice[1:]]] -= distance_dim1_lower
            distance_dim1_upper_slice = np.s_[0, rings[0] + current_ring:, :]
            ideal_grid[distance_dim1_upper_slice][update_allowed[distance_dim1_upper_slice[1:]]] -= distance_dim1_upper

        if (rings[1] - current_ring - 1) >= 0:
            distance_dim2_lower = ideal_grid[1, :, rings[1] - current_ring - 1].mean() - array_view[1, :, 0].mean()
            distance_dim2_upper = ideal_grid[1, :, rings[1] + current_ring].mean() - array_view[1, :, -1].mean()
            distance_dim2_lower_slice = np.s_[1, :, : rings[1] - current_ring]
            ideal_grid[distance_dim2_lower_slice][update_allowed[distance_dim2_lower_slice[1:]]] -= distance_dim2_lower
            distance_dim2_upper_slice = np.s_[1, :, rings[1] + current_ring:]
            ideal_grid[distance_dim2_upper_slice][update_allowed[distance_dim2_upper_slice[1:]]] -= distance_dim2_upper

    for ring in range(rings.max()):

        if (rings[0] - ring - 1) > 0:
            shape_dim1_lower = rings[0] - ring - 1
            shape_dim1_upper = rings[0] + ring + 1
        else:
            shape_dim1_lower = 0
            shape_dim1_upper = shape[0] + 1

        if (rings[1] - ring - 1) > 0:
            shape_dim2_lower = rings[1] - ring - 1
            shape_dim2_upper = rings[1] + ring + 1
        else:
            shape_dim2_lower = 0
            shape_dim2_upper = shape[1] + 1

        update_filter = update_allowed[shape_dim1_lower: shape_dim1_upper, shape_dim2_lower: shape_dim2_upper]
        section = ideal_grid[:, shape_dim1_lower: shape_dim1_upper, shape_dim2_lower: shape_dim2_upper]

        _look_replace(section, update_filter)

        update_filter.fill(0)

        _push_ideal(section, ring)

    return ideal_grid


def build_grid_from_center(x_data, y_data, center, dx, dy, grid_shape, max_sq_dist=105):

    grid0 = (((np.mgrid[0: grid_shape[0], 0: grid_shape[1]]).astype(np.float)
             - np.array(grid_shape).reshape((2, 1, 1)) / 2.0) + 0.5
             ) * np.array((dx, dy)).reshape((2, 1, 1))

    center = np.array(center)

    grid = grid0 + center.reshape((2, 1, 1))

    if x_data is not None and y_data is not None:
        return replace_ideal_with_observed(grid, x_data, y_data, max_sq_dist)
    else:
        return grid


def get_validated_grid(im, grid, delta_dim1, delta_dim2, adjusted_values):

    """Check so grid coordinates actually fall inside image"""

    if grid.shape[0] == 2:
        delta_slice_dim1 = np.s_[0, :, :]
        delta_slice_dim2 = np.s_[1, :, :]
    else:
        delta_slice_dim1 = np.s_[:, :, 0]
        delta_slice_dim2 = np.s_[:, :, 1]

    # If any place has negative numbers, it's an overshoot
    if (grid[delta_slice_dim1].min() < np.ceil(delta_dim1 * 0.5) or
            grid[delta_slice_dim2].min() < np.ceil(delta_dim2 * 0.5)):

        print "*** Invalid grid (less than 0)"
        return grid, True, False

    # If max plus half grid cell is larger than im it's an overshoot too
    if (grid[delta_slice_dim1].max() >= im.shape[0] + np.ceil(delta_dim1 * 0.5) or
            grid[delta_slice_dim2].max() >= im.shape[1] + np.ceil(delta_dim2 * 0.5)):

        print "*** Invalid grid (more than max)"
        return grid, True, False

    return grid, adjusted_values, True


def get_valid_parameters(center, spacing, expected_center, expected_spacing,
                         sigma_spacing=0.55, t=0.95):

    """This function validates observed spacing and center and uses
    expected values when deemed unrealistic.

    The sigma_spacing is set heuristically to reflect acceptable varation.

    The sigma for center is set from mean spacing so that the allowed
    variation relates to the inter-colony distance with a large safty
    margin.
    """

    def get_gauss_probability(height, position, width, data):
        return height * np.exp(-(data - position) ** 2 / (2 * width ** 2))

    print("*** Got center {0} and spacing {1}".format(center, spacing))

    spacing = np.array(spacing, dtype=np.float)
    expected_spacing = np.array(expected_spacing, dtype=np.float)

    p_spacing = get_gauss_probability(1.0, expected_spacing, sigma_spacing, spacing)
    adjust_spacing = p_spacing < t
    spacing[adjust_spacing] = expected_spacing[adjust_spacing]

    center = np.array(center, dtype=np.float)
    expected_center = np.array(expected_center, dtype=np.float)
    sigma_center = 0.5 * spacing.mean()
    p_center = get_gauss_probability(1.0, expected_center, sigma_center, center)
    adjust_center = p_center < t
    center[adjust_center] = expected_center[adjust_center]

    values_adjusted = adjust_center.any() or adjust_spacing.any()

    print('*** Returning center {0} and spacing {1}'.format(center, spacing))

    return tuple(center), tuple(spacing), values_adjusted


def get_grid(
        im,
        expected_spacing=(105, 105),
        grid_shape=(16, 24),
        x_data=None,
        y_data=None,
        expected_center=(100, 100),
        dev_reduce_grid_data_fraction=None,
        validate_parameters=False,
        grid_correction=None):
    """Detects grid candidates and constructs a grid"""

    adjusted_values = True
    center = expected_center
    spacings = expected_spacing

    adaptive_threshold = get_adaptive_threshold(
        im, threshold_filter=None, segments=100, sigma=30)

    im_filtered = get_denoise_segments(im < adaptive_threshold, iterations=3)
    del adaptive_threshold

    if expected_spacing is None:
        expected_spacing = tuple(
            float(a) / b for a, b in zip(im.shape, grid_shape))

    get_segments_by_size(
        im_filtered, min_size=40,
        max_size=expected_spacing[0] * expected_spacing[1], inplace=True)

    get_segments_by_shape(im_filtered, expected_spacing, inplace=True)

    labeled, labels = ndimage.label(im_filtered)
    if x_data is None or y_data is None:
        if labels > 0:
            centra = ndimage.center_of_mass(im_filtered,
                                            labeled, range(1, labels + 1))
            x_data, y_data = np.array(centra).T
            adjusted_values = False

    del labeled

    if dev_reduce_grid_data_fraction is not None:
        filter_grid_data = (
            np.random.random(x_data.shape) < dev_reduce_grid_data_fraction
        )
        x_data = x_data[filter_grid_data]
        y_data = y_data[filter_grid_data]

    if adjusted_values is False:

        center, spacings = get_grid_parameters(
            x_data, y_data, grid_shape, spacings=expected_spacing)

        if center is None or spacings is None:
            return None, x_data, y_data, center, spacings, adjusted_values

        if grid_correction is not None:
            center = tuple(a + b for a, b in
                           zip(center, [i * j for i, j in
                                        zip(grid_correction, spacings)]))

            adjusted_values = True

        if validate_parameters:
            center, spacings, adjusted_values = get_valid_parameters(
                center, spacings, expected_center, expected_spacing)

    dx, dy = spacings

    grid = build_grid_from_center(x_data, y_data, center, dx, dy, grid_shape)

    return grid, x_data, y_data, center, spacings, adjusted_values
