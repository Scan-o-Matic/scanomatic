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

#
# SCANNOMATIC LIBRARIES
#

#import src.resource_signal as resource_signal

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

            T[ndimage.binary_dilation(labled == l, iterations=4)] = \
                 threshold_filter(im[labled == l], *args, **kwargs)

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

    erode_im = ndimage.binary_erosion(im, **kwargs)
    reconstruct_im = ndimage.binary_propagation(erode_im, mask=im)
    tmp = np.logical_not(reconstruct_im)
    erode_tmp = ndimage.binary_erosion(tmp, **kwargs)
    reconstruct_final = np.logical_not(ndimage.binary_propagation(
        erode_tmp, mask=tmp))

    return reconstruct_final


def get_segments_by_size(im, min_size, max_size=-1, inplace=True):

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


def demo(im, box_size=(105, 105), visual=True):

    T = get_adaptive_threshold(im, threshold_filter=None, segments=70, 
        sigma=None)

    im_filtered = get_denoise_segments(im<T, iterations=3)

    get_segments_by_size(im_filtered, min_size=40,
        max_size=box_size[0]*box_size[1], inplace=True)

    labled, labels = ndimage.label(im_filtered)
    centra = ndimage.center_of_mass(im_filtered, labled, range(1, labels+1))
    X, Y = np.array(centra).T

    if visual:
        from matplotlib import pyplot as plt
        plt.imshow(im_filtered)
        plt.plot(Y, X, 'g+', ms=10, mew=2)
        plt.ylim(0, im_filtered.shape[0])
        plt.xlim(0, im_filtered.shape[1])

    return X, Y, labled
