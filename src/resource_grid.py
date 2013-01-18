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

        if (labled==l).sum() > 1:

            if im[labled == l].std() !=0:

                T[ndimage.binary_dilation(labled == l, iterations=4)] = \
                    threshold_filter(im[labled == l], *args, **kwargs)

            else:

                T[ndimage.binary_dilation(labled == l, iterations=4)] = \
                    im[labled == l].mean()

    return ndimage.gaussian_filter(T, sigma=sigma)


def _get_sectioned_image(im):
    """Sections image in proximity regions for points of interests"""

    d = ndimage.distance_transform_edt(im==0)
    k = np.array([[-1, 2, -1]])
    d2 = ndimage.convolve(d, k) + ndimage.convolve(d, k.T)
    d2 = ndimage.binary_dilation(d2 > d2.mean(), border_value=1) == 0
    labled, labels = ndimage.label(d2)

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

    out[remove_pixels] = 0

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
        x, y = find_valid(x, y)

        #put in grid
        #print coord, grid[coord].shape
        grid[coord] = np.array((x, y)).reshape(grid[coord].shape)

        first_loop = False

    return grid


def get_grid(im, box_size=(105, 105), grid_shape=(16, 24), visual=False, X=None, Y=None):
    """Detects grid candidates and constructs a grid"""

    T = get_adaptive_threshold(im, threshold_filter=None, segments=70, 
        sigma=None)

    im_filtered = get_denoise_segments(im<T, iterations=3)
    del T

    get_segments_by_size(im_filtered, min_size=40,
        max_size=box_size[0]*box_size[1], inplace=True)

    labled, labels = ndimage.label(im_filtered)
    if X is None or Y is None:
        centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
        X, Y = np.array(centra).T

    del labled

    x_offset, y_offset, dx, dy = get_grid_parameters(X, Y, 
        expected_distance=box_size[0], grid_shape=grid_shape,
        leeway=1.1)

    grid = build_grid(X, Y, x_offset, y_offset, dx, dy, grid_shape=grid_shape,
        square_distance_threshold=70)

    if visual:
        from matplotlib import pyplot as plt
        plt.imshow(im_filtered)
        plt.plot(Y, X, 'g+', ms=10, mew=2)
        plt.plot(grid[:,:,1].ravel(), grid[:,:,0].ravel(),
            'o', ms=15, mec='w', mew=2, mfc='none')
        plt.ylim(0, im_filtered.shape[0])
        plt.xlim(0, im_filtered.shape[1])
        plt.show()

    return grid
