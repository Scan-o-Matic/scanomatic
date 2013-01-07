#!/usr/bin/env python
"""Some pre-processing techniques"""

import numpy as np
import itertools
from scipy import ndimage
from numpy.lib import stride_tricks

def get_rotating_kernels():

    kernels = list()

    protokernel = np.arange(9).reshape(3,  3)

    for k in xrange(9):

        ax1, ax2 = np.where(protokernel==k)
        kernel = np.zeros((5,5), dtype=bool)
        kernel[ax1: ax1+3, ax2: ax2+3] = 1
        kernels.append(kernel)

    return kernels


def get_rotation_smooth2(im, **kwargs):

    windows_shape = (3, 3, 3, 3)
    section_strides = (im.itemsize * 5, im.itemsize)
    windows_strides = section_strides + section_strides

    _as_strided=stride_tricks.as_strided

    def rotation_matrix2(section):

        windows = _as_strided(section, windows_shape, windows_strides)
        rot_filters = windows.reshape(9, 9)
        return rot_filters[rot_filters.var(1).argmin(),:].mean()

    return ndimage.filters.generic_filter(im, rotation_matrix2, size=5, **kwargs)

def get_rotation_smooth(im, **kwargs):

    kernels = np.array([k.ravel() for k in get_rotating_kernels()],
                dtype=bool)

    def rotation_matrix(section):

        multi_s = stride_tricks.as_strided(section, shape=(9,25),
            strides=(0, section.itemsize))

        rot_filters = multi_s[kernels].reshape(9,9)

        return rot_filters[rot_filters.var(1).argmin(),:].mean()

    return ndimage.filters.generic_filter(im, rotation_matrix, size=5, **kwargs)


def get_histogram_equalization(im):
    """Performs histogram equalization"""

    #SET DATA ABOUT IMAGE
    G = im.max() + 1  # Grayscale is assumed to start at 0
    NM = im.size
    a = (G - 1.0) / NM

    H =  np.histogram(im, bins=np.arange(0, G+1))[0]
    Hc = H.cumsum()  # Culmulative histogram

    #Transformation array such that the resulting image will
    #have a equal distribution of brightnesses
    T = np.round(Hc * a).astype(np.int)

    #For each value in im use the value at that index in T
    return np.take(T, im.astype(np.int))


def get_translocation_4pt(A, B, to_origo=True):

    dA = A.max(1)
    dB = B.max(1)

    if to_origo:
        mv1 = (0, 0) - B[dB.argmin(), :]
    else:
        mv1 = (0, 0)

    mv2 = B[dB.argmin(), :] - A[dA.argmin(), :]

    return A + mv1 + mv2, B + mv1


def get_distances_and_pt_mapping(A, B):

    dists = list()
    perms = list()
    tots = list()

    for p in itertools.permutations(xrange(4)):
        dists.append(np.abs((B - A[p, :])).max(1))
        tots.append(dists[-1].sum())
        perms.append(p)
        

    p_index = np.array(tots).argmin()

    return dists[p_index], perms[p_index]


def get_rotation(A, B, perm=None):

    A, B = get_translocation_4pt(A, B)

    if perm is None:

        dists, perm = get_distances_and_pt_mapping(A, B)

    #Select a second point from the others, unique is the one one
    #the diagonal

    L = np.sqrt((A**2).sum(1))

    pt_a = L.argmax()
    pt_b = perm[pt_a]

    theta_a = np.arctan2(*A[pt_a, :])
    theta_b = np.arctan2(*B[pt_b, :])

    return theta_b - theta_a
