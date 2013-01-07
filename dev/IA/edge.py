
import scipy.ndimage as scind
import numpy as np

#
# GLOBALS
#


#
# HIDDEN
#

def _has_zero_cross(a, b):

    return (a > 0) and (b < 0)

#
# PUBLIC
#

def get_edges(im, sigma=2.0, local_context=2):

    #First get the smooth 2nd derivative
    LoG = scind.gaussian_laplace(input=im, sigma=sigma)

    LoG_max = scind.filters.maximum_filter(LoG, local_context)
    LoG_min = scind.filters.minimum_filter(LoG, local_context)

    v_Z = np.frompyfunc(_has_zero_cross, 2, 1)

    return v_Z(LoG_max, LoG_min)
    
