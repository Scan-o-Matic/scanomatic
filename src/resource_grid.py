#!/usr/bin/env python
"""
This module contains gridding specific resources.
"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
from skimage import filter as ski_filter
from scipy import ndimage
import types
from collections import defaultdict
import functools
from scipy.optimize import fsolve

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

        print "*** Done label", l

    print "*** Will smooth it out"
    return ndimage.gaussian_filter(T, sigma=sigma)


def _get_sectioned_image(im):
    """Sections image in proximity regions for points of interests"""

    print "*** Ready to section"
    d = ndimage.distance_transform_edt(im==0)
    print "*** Distances made"
    k = np.array([[-1, 2, -1]])
    d2 = ndimage.convolve(d, k) + ndimage.convolve(d, k.T)
    print "*** Edges detected"
    d2 = ndimage.binary_dilation(d2 > d2.mean(), border_value=1) == 0
    print "*** Areas defined"
    labled, labels = ndimage.label(d2)
    print "*** Areas labled"
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


def get_grid_parameters(X, Y, expected_distance=105, grid_shape=(16, 24),
    leeway=1.1, expected_start=(100, 100)):
    """Gets the parameters of the ideal grid based on detected candidate
    intersects.

    returns x_offset, y_offset, dx, dy

    where offsets describe coordinates for position (0, 0) of the grid.
    """

    if X.size == 0:

        return expected_start[0] , expected_start[1], expected_distance, expected_distance

    #Calculate row/column distances
    XX = X.reshape((1, X.size))
    dX = np.abs(XX - XX.T).reshape((1, X.size ** 2))
    dxs = dX[np.where(np.logical_and(dX > expected_distance / leeway,
                                     dX < expected_distance * leeway))]
    if dxs.size == 0:
        dx = expected_distance
        x_offset = expected_start[0]
    else:
        dx = dxs.mean()

        #Calculate the offsets
        Ndx = np.array([np.arange(grid_shape[0])]) * dx
        x_offsets = XX - Ndx.T
        x_offset = np.median(x_offsets)

    YY = Y.reshape((1, Y.size))
    dY = np.abs(YY - YY.T).reshape((1, Y.size ** 2))
    dys = dY[np.where(np.logical_and(dY > expected_distance / leeway,
                                     dY < expected_distance * leeway))]

    if dys.size == 0:
        dy = expected_distance
        y_offset = expected_start[0]
    else:
        dy = dys.mean()

        #Calculate the offsets
        Ndy = np.array([np.arange(grid_shape[1])]) * dy
        y_offsets = YY - Ndy.T
        y_offset = np.median(y_offsets)

    return x_offset, y_offset, dx, dy


def get_grid_parameters_3(X, Y, im, expected_distance=105, 
    grid_shape=(16, 24), leeway=0.1, expected_start=(100, 100)):

    dx, dy = get_grid_spacings(X, Y, expected_dx, expected_dy, leeway=leeway)

    pos_list = np.arange(X.size)

    rowL = grid_shape[0] ** 2 * grid_shape[1] / float(X.size)
    colL = grid_shape[1] ** 2 * grid_shape[0] / float(Y.size)

    #Guess what row (only works for evenly detected...
    Xis = (pos_list[X.argsort()] / colL).astype(np.int)
    Yis = (pos_list[Y.argsort()] / rowL).astype(np.int)

    #Midpoint:
    mx = grid_shape[1] / 2.0
    my = grid_shape[0] / 2.0

    dXis = (Xis - mx) * dx
    dYis = (Yis - my) * dy

    #Vote
    votes = np.zeros(im.shape)
    for pos in xrange(X.size):
        votes[np.int(np.round(X[pos] + dXis)),
            np.int(np.round(Y[pos] + dYis))] += 1

    votes = ndimage.gaussian_filter(votes, sigma=(dx + dy)/2*leeway)

    return np.unravel_index(votes.argmax(), votes.shape), dx, dy


def get_grid_spacings(X, Y, expected_dx, expected_dy, leeway=0.1):

    dXs = np.abs(np.subtract.outer(X, X))
    dx = dXs[np.logical_and(dXs > expected_dx * (1 - leeway),
            dXs < expected_dx * (1 + leeway))].mean()

    dYs = np.abs(np.subtract.outer(Y, Y))
    dy = dYs[np.logical_and(dYs > expected_dy * (1 - leeway),
            dYs < expected_dy * (1 + leeway))].mean()

    return dx, dy


def get_prior(H, I=None):

    if I == None:
        Prior = np.ones(H.shape[1:], dtype=np.float16)
    else:
        Hx, Hy = H
        x0, y0, sx, sy, ns = I
        N = 2 * np.pi * sx * sy * ns ** 2  # this is REALLY not important!
        Prior = np.exp(-0.5 * ((Hx - x0) ** 2 / (ns * sx) ** 2 + \
                               (Hy - y0) ** 2 / (ns * sy) ** 2)) / N

    return Prior


def get_likelyhood(H, X, Y, I0, S, I=None):
    """
    This is the most important function. It describes how likely/expected the
    actual Data is, given a certain hypothesis H.

    The assumptions of this particular Likelyhood function is that the grid is
    equally spaced, has no rotation, and that the measured grid points will be
    Normal/Gaussian distributed around the real grid points with a spread
    S=(sx, sy). Further, it is assumed that X and Y jitter is uncorrelated,
    and of equal magnitude for all data points.

    For sx and sy, a good value to start with is probably 1 or 2 times the
    leeway * (dx, dy), as defined by get_spacing().
    """

    Hx, Hy = H  # These are 1D arrays 
    grid_X, grid_Y = I0.reshape((2, I0.shape[1] * I0.shape[2]))  # Thesse too
    sx, sy = S  # Scalars

    grid_Hx = np.add.outer(Hx, grid_X)
    grid_Hy = np.add.outer(Hy, grid_Y)

    N = X.size * grid_X.size * 2 * np.pi * S.prod()  # still not important...
    del I0, grid_X, grid_Y

    print "Attempting array size", X.size * grid_Hx.size

    L = np.exp(-0.5 * 
        (np.subtract.outer(X, grid_Hx, dtype=np.float16) ** 2 / sx ** 2 + 
        np.subtract.outer(Y, grid_Hy, dtype=np.float16) ** 2 / sy ** 2)) / N

    return L.sum(axis=0).sum(axis=1)

def get_vectorized_prototypes(X, Y, I0, S, I=None):

    grid_X, grid_Y = I0.reshape((2, I0.shape[1] * I0.shape[2]))  # arrays
    sx, sy = S  # Scalars

    X_grid = np.subtract.outer(X, grid_X)
    Y_grid = np.subtract.outer(Y, grid_Y)

    N = X.size * grid_X.size * 2 * np.pi * S.prod()  # still not important...

    return X_grid, Y_grid, N

def get_likelyhood_vectorized(hx, hy, X_grid, Y_grid, sx, sy, N, I=None):

    L = np.exp(-0.5 * 
        (X_grid + hx) ** 2 / sx ** 2 + 
        (Y_grid + hy) ** 2 / sy ** 2) / N

    return L.sum()


def dev_get_grid_parameters(X, Y, expected_distance=54, grid_shape=(32, 48),
    leeway=0.1, expected_start=(100, 100), im_shape=None, ns=1.0):

    D = np.array(get_grid_spacings(X, Y, expected_distance, expected_distance,
        leeway))

    dx = D[0]
    dy = D[1] 

    S = D * ns * leeway / np.sqrt(2)

    #DEFINE SEARCHSPACE
    if im_shape is not None:
        H = np.mgrid[
            np.floor(0.5 * Y.min()): np.floor(im_shape[1] - dy * grid_shape[1]),
            np.floor(0.5 * X.min()): np.floor(im_shape[0] - dx * grid_shape[0])]

    else:
        H = np.mgrid[np.floor(0.5 * Y.min()): 1.1 * np.ceil(Y.mean()): 1,
                          np.floor(0.5 * X.min()): 1.1 * np.ceil(X.mean()): 1]

    H = H.reshape((2, H.shape[1] * H.shape[2])).astype(np.float16)

    print "Hypothesis", H.min(axis=1), 'to', H.max(axis=1)

    #DEFINE IDEAL GRID AT ZERO OFFSET
    I0 = (np.mgrid[: grid_shape[1], : grid_shape[0]]).astype(np.float16)
    I0[0, :, :] *= dx
    I0[1, :, :] *= dy


    print "Search space given ideal grid and X, Y:", I0.size * X.size

    #INIT FOR VECTORIZED SEARCHING
    X_grid, Y_grid, N = get_vectorized_prototypes(X, Y, I0, S, I=None)
    partial_get_likelyhood = functools.partial(get_likelyhood_vectorized,
        X_grid=X_grid, Y_grid=Y_grid, sx=S[0], sy=S[1], N=N, I=None)

    vectorized_likelyhood = np.frompyfunc(partial_get_likelyhood, 2, 1)

    #SEARCHING
    P = vectorized_likelyhood(H[0], H[1]) * get_prior(H)

    #GET MOST PROBABLE
    x_offset, y_offset = H[:, P.argmax()]

    return x_offset, y_offset, dx, dy


def replace_ideal_with_observed(iGrid, X, Y, max_sq_dist):
 
    iX = iGrid[0]
    iY = iGrid[1]

    def _get_replacement(x, y):

        D = (X - x)**2 + (Y - y)**2 
        if (D < max_sq_dist).any():
            x = X[D.argmin()]
            y = Y[D.argmin()]

        return x, y
 
    vectorized_replacement = np.frompyfunc(_get_replacement, 2, 2)

    grid = np.array(vectorized_replacement(iX, iY))

    return grid    


def build_grid_from_center(X, Y, center, dx, dy, grid_shape, max_sq_dist=25):

    grid0 = (((np.mgrid[0: grid_shape[0], 0: grid_shape[1]]).astype(np.float)
        - np.array(grid_shape).reshape(2, 1, 1) / 2.0) + 0.5
        ) * np.array((dx, dy)).reshape(2, 1, 1)

    def grid_energy(c, grid0):

        gGrid = grid0 + c.reshape(2, 1, 1)
        gX = gGrid[0].ravel()
        gY = gGrid[1].ravel()
        obs_guess_G = replace_ideal_with_observed(gGrid, X, Y, max_sq_dist)
        ogX = obs_guess_G[0]
        ogY = obs_guess_G[1]

        f = np.logical_or(gX != ogX, gY != ogY)
        return np.log(np.sqrt(np.power((gGrid[f] - obs_guss_G[f]), 2))).sum()

    #Solve grid_energy
    center = fsolve(grid_energy, x0=np.array(center), args=(grid0,))
    grid = grid0 + center.reshape(2, 1, 1)

    return replace_ideal_with_observed(grid, X, Y, max_sq_dist)


def build_grid(X, Y, x_offset, y_offset, dx, dy, grid_shape=(16,24),
    square_distance_threshold=None):
    """Builds grids based on candidates and parameters"""

    if square_distance_threshold is None:
        square_distance_threshold = ((dx + dy) / 2.0 * 0.05) ** 2

    grid = np.zeros(grid_shape + (2,), dtype=np.float)
    
    D = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            D[i,j] = i * (1 + 1.0 / (grid_shape[0] + 1)) + j

    rD = D.ravel().copy()
    rD.sort()

    def find_valid(x, y):

        d = (X - x) ** 2 + (Y - y) ** 2
        valid = d < square_distance_threshold
        if valid.any():
            pos = d == d[valid].min()
            if pos.sum() == 1:
                return X[pos], Y[pos]

        return x, y

    x = x_offset
    y = y_offset
    first_loop = True

    for v in rD:
        #get new position
        coord = np.where(D == v)

        #generate a reference position already passed
        if coord[0][0] > 0:
            old_coord = (coord[0] - 1, coord[1])
        elif coord[1][0] > 0:
            old_coord = (coord[0], coord[1] - 1)

        if not first_loop:
            #calculate ideal step
            x, y = grid[old_coord].ravel()
            x += (coord[0] - old_coord[0]) * dx
            y += (coord[1] - old_coord[1]) * dy

        #modify with observed point close to ideal if exists
        if X.size > 0:
            x, y = find_valid(x, y)

        #put in grid
        #print coord, grid[coord].shape
        grid[coord] = np.array((x, y)).reshape(grid[coord].shape)

        first_loop = False

    return grid


def get_blob_centra(im_filtered):

    labled, labels = ndimage.label(im_filtered)
    if labels > 0:
        centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
        X, Y = np.array(centra).T
    else:
        X = np.array([])
        Y = np.array([])

    return X, Y

def get_grid(im, box_size=(105, 105), grid_shape=(16, 24), visual=False, X=None, Y=None, 
    expected_offset=(100, 100), run_dev=False, dev_filter_XY=None):
    """Detects grid candidates and constructs a grid"""

    print "** Will threshold"

    T = get_adaptive_threshold(im, threshold_filter=None, segments=40, 
        sigma=30)

    print "** Got T"

    im_filtered = get_denoise_segments(im<T, iterations=3)
    del T

    print "** Filtered 1st pass the im<T, removed T"

    get_segments_by_size(im_filtered, min_size=40,
        max_size=box_size[0]*box_size[1], inplace=True)

    print "** Filtered on size"

    get_segments_by_shape(im_filtered, box_size, inplace=True)

    print "** Filtered on shape"

    labled, labels = ndimage.label(im_filtered)
    if X is None or Y is None:
        if labels > 0:
            centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
            X, Y = np.array(centra).T
        else:
            X = np.array([])
            Y = np.array([])

    del labled

    print "** Got X and Y"
    if dev_filter_XY is not None:
        f_XY = np.random.random(X.shape) < dev_filter_XY
        X = X[f_XY]
        Y = Y[f_XY]

    if run_dev:
        x_offset, y_offset, dx, dy = dev_get_grid_parameters(X, Y, 
            expected_distance=box_size[0], grid_shape=grid_shape,
            im_shape=im.shape)
    else:
        x_offset, y_offset, dx, dy = get_grid_parameters(X, Y,
            expected_distance=box_size[0], grid_shape=grid_shape,
            leeway=1.1, expected_start=expected_offset)

    print "** Got grid parameters"

    grid = build_grid(X, Y, x_offset, y_offset, dx, dy, grid_shape=grid_shape,
        square_distance_threshold=70)

    print "** Got grid"

    if visual:
        from matplotlib import pyplot as plt
        plt.imshow(im_filtered)
        plt.plot(Y, X, 'g+', ms=10, mew=2)
        plt.plot(grid[:,:,1].ravel(), grid[:,:,0].ravel(),
            'o', ms=15, mec='w', mew=2, mfc='none')
        plt.ylim(0, im_filtered.shape[0])
        plt.xlim(0, im_filtered.shape[1])
        plt.show()

    return grid, X, Y
