import numpy as np
from scipy import ndimage


def ridge_detection(im, **kwargs):
    """im should be int array that has been
    distance transformed"""

    kernelH = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]).ravel()
    kernelV = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).ravel()
    kernelD = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).ravel()

    def _ridge(section):

        sH = section[kernelH]
        sV = section[kernelV]
        sD = section[kernelD]

        return sum((sH.max() == sH[1], sV.max() == sV[1], sD.max() == sD[1])) > 1

    return ndimage.filters.generic_filter(im, _ridge, size=3, origin=(1,1), **kwargs)
