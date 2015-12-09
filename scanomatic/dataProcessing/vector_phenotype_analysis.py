__author__ = 'martin'

import numpy as np
from itertools import product, izip
from scipy.signal import gaussian

from scanomatic.dataProcessing.growth_phenotypes import Phenotypes


def get_plate_phenotype_in_array(phenotypes, phenotype=Phenotypes.GrowthVelocityVector):

    data = phenotypes[..., phenotype.value]
    vector_length = max(v.size for v in data.ravel())

    arr = np.zeros(data.shape + (vector_length,), dtype=np.float)
    for coord in product(*tuple(range(dim_length) for dim_length in data.shape)):
        arr[coord][:len(data[coord])] = data[coord]

    return arr


def get_linearized_positions(data):

    return np.lib.stride_tricks.as_strided(
        data,
        shape=(data.shape[0] * data.shape[1],) + data.shape[2:],
        strides=(data.strides[1],) + data.strides[2:])


def _resolve_neighbours_gauss(data):

    gauss = gaussian(data.shape[0] * 2, 3)

    for coord in izip(*np.where(np.isfinite(data) == False)):

        data[coord] = np.ma.masked_invalid(data[:, coord[1]] * gauss[data.shape[0] - coord[0]: gauss.shape[0] - coord[0]]).mean()

    return data


def _blank_missing_data(data):

    data[:, np.where(np.isfinite(data).sum(axis=0) == 0)] = 0

    return data


def get_pca_components(data, resolve_nans_method=_resolve_neighbours_gauss, dims=2):

    while data.ndim > 2:
        data = get_linearized_positions(data)

    M = data.T.copy()
    print (np.isfinite(M) == False).sum()
    M = _blank_missing_data(M)
    print (np.isfinite(M) == False).sum()
    M = resolve_nans_method(M)
    print (np.isfinite(M) == False).sum()
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T
    return tuple(s[dim]**(1./2) * V[:,dim] for dim in range(dims))