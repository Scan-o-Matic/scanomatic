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

''' NOT IN USE

def get_grid_parameters(X, Y, expected_distance=105, grid_shape=(16, 24),
    leeway=1.1, expected_start=(100, 100)):
    """Gets the parameters of the ideal grid based on detected candidate
    intersects.

    returns x_offset, y_offset, dx, dy

    where offsets describe coordinates for position (0, 0) of the grid.
    """

    if X.size == 0:

        return expected_start[0] , expected_start[1], expected_distance, expected_distance

    dx, dy = get_grid_spacings(X, Y, expected_distance, expected_distance, leeway=leeway-1)
    #Calculate the offsets
    XX = X.reshape((1, X.size))
    Ndx = np.array([np.arange(grid_shape[0])]) * dx
    x_offsets = (XX - Ndx.T).ravel()
    #x_offsets.sort()
    #x_offset = x_offsets[x_offsets.size/4: -x_offsets.size/4].mean()
    x_offset = np.median(x_offsets)

    YY = Y.reshape((1, Y.size))
    Ndy = np.array([np.arange(grid_shape[1])]) * dy
    y_offsets = (YY - Ndy.T).ravel()
    #y_offsets.sort()
    #y_offset = y_offsets[y_offsets.size/4: -y_offsets.size/4].mean()
    y_offset = np.median(y_offsets)

    return x_offset, y_offset, dx, dy

'''


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

''' NOT IN USE

def get_grid_parameters_3(X, Y, im, expected_distance=105, 
    grid_shape=(16, 24), leeway=0.1, expected_start=None):

    dx, dy = get_grid_spacings(X, Y, expected_distance, expected_distance, leeway=leeway)

    fraction_found = float(X.size) / (grid_shape[0] * grid_shape[1])
    rowL = grid_shape[0] * fraction_found
    colL = grid_shape[1] * fraction_found

    #Guess what row (only works for evenly detected...
    Xis = np.zeros(X.shape)
    Yis = np.zeros(Y.shape)
    Xis[X.argsort()] = np.arange(X.size)
    Yis[Y.argsort()] = np.arange(Y.size)
    Xis = (Xis / colL).astype(np.int)
    Yis = (Yis / rowL).astype(np.int)

    #Midpoint:
    my = grid_shape[1] / 2.0 + 0.5
    mx = grid_shape[0] / 2.0 + 0.5

    dXis = (Xis - mx) * dx
    dYis = (Yis - my) * dy

    #print dXis.min(), dXis.max(), dYis.min(), dYis.max()
    #print X.max(), Y.max()
    #print dXis[X.argmax()], dYis[Y.argmax()]

    #Vote
    votes = np.zeros(im.shape)
    for pos in xrange(X.size):
        try:
            votes[np.round(X[pos] - dXis[pos]).astype(np.int),
                np.round(Y[pos] - dYis[pos]).astype(np.int)] += 1
        except IndexError:
            pass
    votes = ndimage.gaussian_filter(votes, sigma=(dx + dy)/2*leeway)
    center = np.array(np.unravel_index(votes.argmax(), votes.shape))

    plt.imshow(votes)
    plt.plot(center[1], center[0], 'ro', ms=4)
    plt.show()


    if expected_start is None:
        expected_start = np.array(im.shape) / 2.0

    if ((center - expected_start)**2).sum() > 3 * expected_distance**2:
        pass  # center = expected_start

    return center, dx, dy

'''

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


''' NOT IN USE

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

    #print "Attempting array size", X.size * grid_Hx.size

    L = np.exp(-0.5 * 
        (np.subtract.outer(X, grid_Hx, dtype=np.float16) ** 2 / sx ** 2 + 
        np.subtract.outer(Y, grid_Hy, dtype=np.float16) ** 2 / sy ** 2)) / N

    return L.sum(axis=0).sum(axis=1)

'''

''' NOT IN USE

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

'''

''' NOT IN USE

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

    #print "Hypothesis", H.min(axis=1), 'to', H.max(axis=1)

    #DEFINE IDEAL GRID AT ZERO OFFSET
    I0 = (np.mgrid[: grid_shape[1], : grid_shape[0]]).astype(np.float16)
    I0[0, :, :] *= dx
    I0[1, :, :] *= dy


    #print "Search space given ideal grid and X, Y:", I0.size * X.size

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

'''

''' NOT IN USE

def replace_ideal_with_observed_depricated(iGrid, X, Y, max_sq_dist):
 
    iX = iGrid[0]
    iY = iGrid[1]

    def _get_replacement(x, y):

        D = (X - x)**2 + (Y - y)**2 
        # print (D.min(), x, y),
        if (D < max_sq_dist).any():
            x = X[D.argmin()]
            y = Y[D.argmin()]

        return x, y
 
    vectorized_replacement = np.frompyfunc(_get_replacement, 2, 2)

    grid = np.array(vectorized_replacement(iX, iY))

    return grid

'''


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

    return replace_ideal_with_observed(grid, X, Y, max_sq_dist)


''' NOT IN USE

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

'''

""" NOT IN USE

def get_blob_centra(im_filtered):

    labled, labels = ndimage.label(im_filtered)
    if labels > 0:
        centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
        X, Y = np.array(centra).T
    else:
        X = np.array([])
        Y = np.array([])

    return X, Y

"""

def get_validated_grid(im, grid, dD1, dD2):

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
        return None
    
    #If max plus half grid cell is larger than im it's an overshoot too
    if (grid[D1].max() >= im.shape[0] + np.ceil(dD1 * 0.5) or
        grid[D2].max() >= im.shape[1] + np.ceil(dD2 * 0.5)):

        print "*** Invalid grid (more than max)"
        return None

    #Grid is OK
    return grid

def get_grid(im, box_size=(105, 105), grid_shape=(16, 24), visual=False, X=None, Y=None, 
    expected_offset=(100, 100), run_dev=False, dev_filter_XY=None):
    """Detects grid candidates and constructs a grid"""

    #print "** Will threshold"

    T = get_adaptive_threshold(im, threshold_filter=None, segments=100, 
        sigma=30)

    #print "** Got T"

    im_filtered = get_denoise_segments(im<T, iterations=3)
    del T

    #print "** Filtered 1st pass the im<T, removed T"

    get_segments_by_size(im_filtered, min_size=40,
        max_size=box_size[0]*box_size[1], inplace=True)

    #print "** Filtered on size"

    get_segments_by_shape(im_filtered, box_size, inplace=True)

    #print "** Filtered on shape"

    labled, labels = ndimage.label(im_filtered)
    if X is None or Y is None:
        if labels > 0:
            centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
            X, Y = np.array(centra).T
        else:
            X = np.array([])
            Y = np.array([])

    del labled

    #print "** Got X and Y"
    if dev_filter_XY is not None:
        f_XY = np.random.random(X.shape) < dev_filter_XY
        X = X[f_XY]
        Y = Y[f_XY]

    if run_dev:
        """
        x_offset, y_offset, dx, dy = dev_get_grid_parameters(X, Y, 
            expected_distance=box_size[0], grid_shape=grid_shape,
            im_shape=im.shape)

        center, dx, dy =  get_grid_parameters_3(X, Y, im,
            expected_distance=box_size[0], 
            grid_shape=grid_shape, leeway=0.1)

        """
        center, spacings = get_grid_parameters_4(X, Y, grid_shape, spacings=box_size, center=None)
        dx, dy = spacings

        #print "** Got grid parameters"

        grid = build_grid_from_center(X, Y, center, dx, dy, grid_shape)
        #Reshape to fit old scheme
        gX, gY = grid
        grid = np.c_[gX, gY].reshape(grid_shape[0], grid_shape[1], 2, order='A')

    else:

        center, spacings = get_grid_parameters_4(X, Y, grid_shape, spacings=box_size, center=None)
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

    if visual:
        from matplotlib import pyplot as plt
        plt.imshow(im_filtered)
        plt.plot(Y, X, 'g+', ms=10, mew=2)
        plt.plot(grid[:,:,1].ravel(), grid[:,:,0].ravel(),
            'o', ms=15, mec='w', mew=2, mfc='none')
        plt.ylim(0, im_filtered.shape[0])
        plt.xlim(0, im_filtered.shape[1])
        plt.show()

    grid = get_validated_grid(im, grid, dy, dx)

    return grid, X, Y
