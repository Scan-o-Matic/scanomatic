#!/usr/bin/env python
"""
This module contains gridding specific resources.
"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.998"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
from skimage import filter as ski_filter
from scipy import ndimage
#import functools
#from scipy.optimize import fsolve

import matplotlib.pyplot as plt

#
# SCANNOMATIC LIBRARIES
#

#
# FUNCTIONS
#


def get_adaptive_threshold(im, threshold_filter=None, segments=60,
        sigma=None, *args, **kwargs):
    """Gives a 2D surface of threshold based on smoothed local measures"""

    if threshold_filter is None:
        threshold_filter = ski_filter.threshold_otsu
    if sigma is None:
        sigma = np.sqrt(im.size)/5

    if segments is None or segments == 5:
        #HACK
        T = np.zeros(im.shape)
        T[im.shape[0]/4, im.shape[1]/4] = 1
        T[im.shape[0]/4, im.shape[1]*3/4] = 1
        T[im.shape[0]*3/4, im.shape[1]/4] = 1
        T[im.shape[0]*3/4, im.shape[1]*3/4] = 1
        T[im.shape[0]/2, im.shape[1]/2] = 1
    else:
        p = 1 - np.float(segments)/im.size
        T = (np.random.random(im.shape) > p).astype(np.uint8)

    labled, labels = _get_sectioned_image(T)

    for l in range(1, labels + 1):

        l_filter = labled == l
        if (l_filter).sum() > 1:

            i_slice = im[l_filter]
            if i_slice.std() !=0:

                T[ndimage.binary_dilation(l_filter, iterations=4)] = \
                    threshold_filter(i_slice, *args, **kwargs)

            else:

                T[ndimage.binary_dilation(l_filter, iterations=4)] = \
                    i_slice.mean()

    #print "*** Will smooth it out"
    return ndimage.gaussian_filter(T, sigma=sigma)


def _get_sectioned_image(im):
    """Sections image in proximity regions for points of interests"""

    #print "*** Ready to section"
    d = ndimage.distance_transform_edt(im==0)
    #print "*** Distances made"
    k = np.array([[-1, 2, -1]])
    d2 = ndimage.convolve(d, k) + ndimage.convolve(d, k.T)
    #print "*** Edges detected"
    d2 = ndimage.binary_dilation(d2 > d2.mean(), border_value=1) == 0
    #print "*** Areas defined"
    labled, labels = ndimage.label(d2)
    #print "*** Areas labled"
    return labled, labels


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

    labled_im, labels = ndimage.label(im)
    segments = ndimage.find_objects(labled_im)

    bound_d1, bound_d2 = max_shape
    roundness_k = np.pi / 4.0
    roundness_t1 = 0.25
    roundness_t2 = 0.1

    for i, segment in enumerate(segments):

        s_d1, s_d2 = segment
        if (abs(s_d1.stop - s_d1.start) > bound_d1 or
            abs(s_d2.stop - s_d2.start) > bound_d2):

            out[segment][labled_im[segment] == i + 1] = False

        elif check_roundness:

            s = im[segment]
            blob = s[labled_im[segment] == i + 1]

            #CHECK IF a) The outer shape is square
            # b) The blob area is close to expected value given
            # outer size
            if (abs(1 - float(s.shape[0]) / s.shape[1]) > roundness_t1 or
                abs(1 - s.size * roundness_k / blob.sum()) > roundness_t2):

                out[segment][labled_im[segment] == i + 1] = False

    if not inplace:
        return out


def get_grid_parameters_4(X, Y, grid_shape, spacings=(54, 54), center=None):

    #grid_shape = (grid_shape[1], grid_shape[0])

    data = (X, Y)
    new_spacings = get_grid_spacings(X, Y, *spacings)
    centers = get_centre_candidates(grid_shape, new_spacings)
    votes = get_votes(data, centers)
    weights = get_weights(votes, data, 1.0)
    sigma = np.max((spacings[0], new_spacings[1])) * 0.1 / np.sqrt(2) + 0.5
    heatmap = get_heatmap(data, votes, weights, sigma)

    _cD2, _cD1 = np.unravel_index(heatmap.argmax(), heatmap.shape)

    new_center = (_cD1, _cD2)

    """
    plt.imshow(heatmap)
    plt.plot(new_center[1], new_center[0], 'ro', ms=4)
    plt.show()
    """
    return new_center, new_spacings


def get_weights(votes, data, width=1):
    """
    Get weights for votes. If width > 0, a Gaussian weight is assigend based
    on the distance of the vote to the mean of the data. The width of the
    Gaussian is set from the spread of the data, and scaled by the width
    parameter. By default, width is set to be 1, which is a very weak
    weighting.
    """

    X, Y = data
    VX, VY = votes

    if width > 0:
        weights = (np.exp(-((VX - X.mean()) ** 2 / (width * X.std()) ** 2 +
                  (VY - Y.mean()) ** 2 / (width * Y.std()) ** 2)) /
                  (2 * np.pi * X.std() * Y.std() * width ** 2))
    else:
        weights = np.ones(VX.shape)

    return weights


def get_votes(data, centres):
    """
    Get votes from all data points.
    """

    X, Y = data
    GX, GY = centres

    VX = np.add.outer(X, GX)
    VY = np.add.outer(Y, GY)

    return VX.ravel(), VY.ravel()


def get_heatmap(data, votes, weights, sigma):
    """
    Get smoothed histogram.

    A good value for sigma is probably  max(dx, dy) * leeway /sqrt(2) + 0.5.
    """

    X, Y = data
    VX, VY = votes

    vote_slice = np.logical_and(np.logical_and(VX >= 0, VY >= 0),
                np.logical_and(VX <= X.max(), VY <= Y.max()))

    votes_x = VX[vote_slice]
    votes_y = VY[vote_slice]
    W = weights[vote_slice]

    heatmap = np.zeros((np.ceil(Y.max()) + 1, np.ceil(X.max()) + 1))

    votes_x = np.round(votes_x).astype(np.int)
    votes_y = np.round(votes_y).astype(np.int)

    #This ravels the indices to match a raveled heatmap
    flat_votes_xy = votes_y * heatmap.shape[1] + votes_x

    #Get unique coordinates and sort
    unique_votes = np.unique(flat_votes_xy)
    unique_votes.sort()

    #Make weighted histogram (the returning array will match the
    #sorted unique_votes, +0.5 is OK since we know indices will be
    #ints and the lowest be 0 (thus -1 is also safe)
    unique_vote_weights = np.histogram(flat_votes_xy, bins=np.hstack(((-1,),
        unique_votes)) + 0.5, weights=W)[0]

    #print np.c_[unique_votes, unique_vote_weights].T

    #Assign the weighted votes
    heatmap.ravel()[unique_votes] = unique_vote_weights

    if sigma > 0:
        heatmap = ndimage.gaussian_filter(heatmap, sigma)

    heatmap /= heatmap.sum()

    return heatmap


def get_centre_candidates(grid_size, spacings):
    """
    Get coordinates of all possible grid centres without offset.
    """

    dx, dy = spacings

    GY, GX = np.mgrid[-(grid_size[1] - 1) * 0.5: (grid_size[1] - 1) * 0.5 + 1,
                      -(grid_size[0] - 1) * 0.5: (grid_size[0] - 1) * 0.5 + 1]

    return GX.ravel() * dx, GY.ravel() * dy


def get_grid_spacings(X, Y, expected_dx, expected_dy, leeway=0.1):

    dXs = np.abs(np.subtract.outer(X, X))
    dx = dXs[np.logical_and(dXs > expected_dx * (1 - leeway),
            dXs < expected_dx * (1 + leeway))].mean()

    dYs = np.abs(np.subtract.outer(Y, Y))
    dy = dYs[np.logical_and(dYs > expected_dy * (1 - leeway),
            dYs < expected_dy * (1 + leeway))].mean()

    return dx, dy


def replace_ideal_with_observed(iGrid, X, Y, max_sq_dist):

    shape = np.array(iGrid.shape[1:])
    gUpdated = np.ones(shape, dtype=np.bool)

    rings = shape / 2

    def _look_replace(array_view, filt):

        iX = array_view[0].ravel()
        iY = array_view[1].ravel()
        D = np.subtract.outer(iX, X) ** 2 + np.subtract.outer(iY, Y) ** 2
        rPos = D.min(axis=1) <= max_sq_dist
        minD = D.argmin(axis=1)[rPos]
        where_pos = np.unravel_index(np.where(rPos)[0], filt.shape)
        for i in xrange(where_pos[0].size):
            d1 = where_pos[0][i]
            d2 = where_pos[1][i]
            if filt[d1, d2]:
                #print "r {0}: {1} -> {2}".format((d1, d2), sect[:,d1, d2], (X[minD[i]], Y[minD[i]]))
                sect[0, d1, d2] = X[minD[i]]
                sect[1, d1, d2] = Y[minD[i]]

    def _push_ideal(array_view, r):

        if (rings[0] - r - 1) >= 0:
            dLD1 = iGrid[0, rings[0] - r - 1, :].mean() - array_view[0, 0, :].mean()
            dUD1 = iGrid[0, rings[0] + r, :].mean() - array_view[0, -1, :].mean()
            lD1slice = np.s_[0, : rings[0] - r, :]
            iGrid[lD1slice][gUpdated[lD1slice[1:]]] -= dLD1
            uD1slice = np.s_[0, rings[0] + r: , :]
            iGrid[uD1slice][gUpdated[uD1slice[1:]]] -= dUD1
            #print dLD1, dUD1, gUpdated[lD1slice[1:]].sum(), gUpdated[uD1slice[1:]].sum()

        if (rings[1] - r - 1) >= 0:
            dLD2 = iGrid[1, :, rings[1] - r - 1].mean() - array_view[1, :, 0].mean()
            dUD2 = iGrid[1, :, rings[1] + r].mean() - array_view[1, :, -1].mean()
            lD2slice = np.s_[1, :, : rings[1] - r]
            iGrid[lD2slice][gUpdated[lD2slice[1:]]] -= dLD2
            uD2slice = np.s_[1, :, rings[1] + r:]
            iGrid[uD2slice][gUpdated[uD2slice[1:]]] -= dUD2
            #print dLD2, dUD2

    for r in xrange(rings.max()):

        if (rings[0] - r - 1) > 0:
            s1L = rings[0] - r - 1
            s1U = rings[0] + r + 1
        else:
            s1L = 0
            s1U = shape[0] + 1

        if (rings[1] - r - 1) > 0:
            s2L = rings[1] - r - 1
            s2U = rings[1] + r + 1
        else:
            s2L = 0
            s2U = shape[1] + 1

        filt = gUpdated[s1L: s1U, s2L: s2U]
        sect = iGrid[:, s1L: s1U, s2L: s2U]

        _look_replace(sect, filt)

        filt.fill(0)

        _push_ideal(sect, r)

    return iGrid

def build_grid_from_center(X, Y, center, dx, dy, grid_shape, max_sq_dist=105):

    grid0 = (((np.mgrid[0: grid_shape[0], 0: grid_shape[1]]).astype(np.float)
        - np.array(grid_shape).reshape(2, 1, 1) / 2.0) + 0.5
        ) * np.array((dx, dy)).reshape(2, 1, 1)

    """

    grid0 = np.mgrid[(-grid_shape[0]/2.0 + 0.5) * dx:
                    (grid_shape[0]/2.0 + 0.5) * dx: dx,
                    (-grid_shape[1]/2.0 + 0.5) * dy:
                    (grid_shape[1]/2.0 + 0.5) * dy: dy]

    """
    center = np.array(center)

    #print "***Building grid from center", center
    '''
    def grid_energy(c, grid0):

        gGrid = grid0 + c.reshape(2, 1, 1)
        #print gGrid[0].min(), gGrid[0].max(), gGrid[1].min(), gGrid[1].max()
        #print X.min(), X.max(), Y.min(), Y.max()
        gX = gGrid[0].ravel()
        gY = gGrid[1].ravel()
        obs_guess_G = replace_ideal_with_observed(gGrid, X, Y, max_sq_dist)
        ogX = obs_guess_G[0]
        ogY = obs_guess_G[1]

        f = np.logical_or(gX != ogX, gY != ogY)
        dG = np.power(gGrid[f] - obs_guess_G[f], 2)
        #print f

        if dG.any() == False:
            return 0

        return np.sqrt(dG).sum()

    #print "***Will improve center {0}".format(center)
    #Solve grid_energy
    #center = fsolve(grid_energy, x0=np.array(center), args=(grid0,))
    '''

    grid = grid0 + center.reshape(2, 1, 1)
    #print "***Will move ideal to observed center"

    if X is not None and Y is not None:
        return replace_ideal_with_observed(grid, X, Y, max_sq_dist)
    else:
        return grid


def get_validated_grid(im, grid, dD1, dD2, adjusted_values):

    """Check so grid coordinates actually fall inside image"""

    #Work with both types of grids
    if grid.shape[0] == 2:
        D1 = np.s_[0,:,:]
        D2 = np.s_[1,:,:]
    else:
        D1 = np.s_[:,:,0]
        D2 = np.s_[:,:,1]

    #If any place has negative numbers, it's an overshoot
    if (grid[D1].min() < np.ceil(dD1 * 0.5) or
        grid[D2].min() < np.ceil(dD2 * 0.5)):

        print "*** Invalid grid (less than 0)"
        return None, True

    #If max plus half grid cell is larger than im it's an overshoot too
    if (grid[D1].max() >= im.shape[0] + np.ceil(dD1 * 0.5) or
        grid[D2].max() >= im.shape[1] + np.ceil(dD2 * 0.5)):

        print "*** Invalid grid (more than max)"
        return None, True

    #Grid is OK
    return grid, adjusted_values


def get_valid_parameters(center, spacing, expected_center, expected_spacing,
        sigma_spacing=0.55, t=0.95):

    """This function validates observed spacing and center and uses
    expected values when deemed unrealistic.

    The sigma_spacing is set heuristically to reflect acceptable varation.

    The sigma for center is set from mean spacing so that the allowed
    variation relates to the inter-colony distance with a large safty
    margin.
    """

    #Gauss function
    def _get_p(a, b, c, x):
        """a is height, b is position, c is width"""
        p = a * np.exp(-(x - b) ** 2 / (2 * c ** 2))
        return p

    print "*** Got center {0} and spacing {1}".format(center, spacing)

    spacing = np.array(spacing)
    expected_spacing = np.array(expected_spacing)

    p_spacing = _get_p(1.0, expected_spacing, sigma_spacing, spacing)
    spacing[p_spacing < t] = expected_spacing[p_spacing < t]

    center = np.array(center)
    expected_center = np.array(expected_center)
    sigma_center = 0.5 * spacing.mean()
    p_center = _get_p(1.0, expected_center, sigma_center, center)
    center[p_center < t] = expected_center[p_center < t]

    adjusted_values = (p_center < t).any() or (p_spacing < t).any()

    print "*** Returning center {0} and spacing {1}".format(center, spacing)

    return tuple(center), tuple(spacing), adjusted_values


def get_grid(im, expected_spacing=(105, 105), grid_shape=(16, 24),
    visual=False, X=None, Y=None,
    expected_center=(100, 100), run_dev=False, dev_filter_XY=None,
    validate_parameters=False):
    """Detects grid candidates and constructs a grid"""

    #print "** Will threshold"
    adjusted_values = False

    T = get_adaptive_threshold(im, threshold_filter=None, segments=100,
        sigma=30)

    #print "** Got T"

    im_filtered = get_denoise_segments(im<T, iterations=3)
    del T

    #print "** Filtered 1st pass the im<T, removed T"

    get_segments_by_size(im_filtered, min_size=40,
        max_size=expected_spacing[0]*expected_spacing[1], inplace=True)

    #print "** Filtered on size"

    get_segments_by_shape(im_filtered, expected_spacing, inplace=True)

    #print "** Filtered on shape"

    labled, labels = ndimage.label(im_filtered)
    if X is None or Y is None:
        if labels > 0:
            centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
            X, Y = np.array(centra).T
        else:

            #IF cant detect any potential colonies anywhere then stop
            center = expected_center
            spacings = expected_spacing
            adjusted_values = True

    del labled

    #print "** Got X and Y"
    if dev_filter_XY is not None:
        f_XY = np.random.random(X.shape) < dev_filter_XY
        X = X[f_XY]
        Y = Y[f_XY]

    if adjusted_values == False:

        if run_dev:
            """
            x_offset, y_offset, dx, dy = dev_get_grid_parameters(X, Y,
                expected_distance=box_size[0], grid_shape=grid_shape,
                im_shape=im.shape)

            center, dx, dy =  get_grid_parameters_3(X, Y, im,
                expected_distance=box_size[0],
                grid_shape=grid_shape, leeway=0.1)

            """
            center, spacings = get_grid_parameters_4(X, Y, grid_shape,
                spacings=expected_spacing, center=None)

            if validate_parameters:
                center, spacings, adjusted_values = get_valid_parameters(center,
                    spacings, expected_center, expected_spacing)
            else:
                adjusted_values = False

        else:

            center, spacings = get_grid_parameters_4(X, Y, grid_shape,
                spacings=expected_spacing, center=None)

            if validate_parameters:
                center, spacings, adjusted_values = get_valid_parameters(center,
                    spacings, expected_center, expected_spacing)
            else:
                adjusted_values = False

    dx, dy = spacings

    #print "** Got grid parameters"

    grid = build_grid_from_center(X, Y, center, dx, dy, grid_shape)
    #Reshape to fit old scheme
    gX, gY = grid
    grid = np.c_[gX, gY].reshape(grid_shape[0], grid_shape[1], 2, order='A')

    """
    x_offset, y_offset, dx, dy = get_grid_parameters(X, Y,
        expected_distance=box_size[0], grid_shape=grid_shape,
        leeway=1.1, expected_start=expected_offset)

    #print "** Got grid parameters"

    grid = build_grid(X, Y, x_offset, y_offset, dx, dy, grid_shape=grid_shape,
        square_distance_threshold=70)

    """

    #print "** Got grid"

    if visual and X is not None and Y is not None:
        from matplotlib import pyplot as plt
        plt.imshow(im_filtered)
        plt.plot(Y, X, 'g+', ms=10, mew=2)
        plt.plot(grid[:,:,1].ravel(), grid[:,:,0].ravel(),
            'o', ms=15, mec='w', mew=2, mfc='none')
        plt.ylim(0, im_filtered.shape[0])
        plt.xlim(0, im_filtered.shape[1])
        plt.show()

    grid, adjusted_values = get_validated_grid(im, grid, dy, dx, adjusted_values)

    return grid, X, Y, center, spacings, adjusted_values
