import numpy as np
from skimage import filter
from scipy import ndimage
import types

def get_sectioned_image(im):
    """Sections image in proximity regions for points of interests"""

    d = ndimage.distance_transform_edt(im==0)
    k = np.array([[-1, 2, -1]])
    d2 = ndimage.convolve(d, k) + ndimage.convolve(d, k.T)
    d2 = ndimage.binary_dilation(d2 > d2.mean(), border_value=1) == 0
    labled, labels = ndimage.label(d2)

    return labled, labels


def get_iterative_threshold(im, filt=None):
    """Iteratively refined threshold to stably section bg/object, needs initial
    guess. If not supplied becomes the four corner pixels"""

    if filt is None:
        filt = np.zeros(im.shape, dtype=np.bool)
        filt[0, 0] = 1
        filt[0, -1] = 1
        filt[-1, 0] = 1
        filt[-1, -1] = 1

    oldT = -1
    T = -2

    while oldT != T:

        oldT = T
        O = im[filt]
        B = im[filt == False]
        T = (O.sum() / np.float(O.size) + B.sum() / np.float(B.size)) / 2.0
        filt = im > T

    return T


def get_p_tile_threshold(im, p, comparison='greater'):
    """Sets a threshold based on a priory knowledge of how large fraction
    of image should fullfill the threshold condition"""

    if comparison == 'greater':
        c = np.greater
        t = 256
        dt = -1
    else:
        c = np.less
        t = 0
        dt = 1

    p *= im.size

    while c(im, t).sum() < p:

        t += dt

    if abs((im < t).sum() - p) < abs((im < t - 1).sum() - p):
 
        return t

    else:

        return t - 1

def _get_context(c=8):

    im = np.zeros((3, 3))

    if c == 8:
        im[:,:] = 1
        im[1,1] = 0
    if c == 4:
        im[1,0] = 1
        im[0,1] = 1
        im[2,1] = 1
        im[1,2] = 1

    return im


class _F_Wrapper(object):

    def __init__(self, f, arg2):

        self.f = f
        self.arg2 = arg2

    def __call__(self, arg1):

        return self.f(arg1, self.arg2)


def get_hysteresis_segmentation(im, t1=None, t1_kwargs={},
    t2=None, t2_kwargs={}, context=8, comparison=None,
    origin=(1,1)):
    """t1 and t2 are either functions or threshold values"""

    #Get context matrix
    if type(context) == types.IntType:
        context = _get_context(context)

    #10 is an arbitrary number large enough to make it different
    context[origin] = context.size * 10

    #Set t1 and t2 as functions if they where just values
    if comparison is None or type(comparsion) != types.FunctionType:

        if comparison == "greater":
            comparison = np.greater
        else:
            comparison = np.less

    if type(t1) == types.FunctionType:
        t1 = t1(im, **t1_kwargs)
    if type(t2) == types.FunctionType:
        t2 = t2(im, **t2_kwargs)

    t1_im = comparison(im, t1)
    t2_im = comparison(im, t2)
    t2_im[t1_im] = 0

    ret_im = np.zeros(t1_im.shape)

    while (ret_im-t1_im).any():

        ret_im[:,:] = t1_im[:,:]
        c_eval = ndimage.convolve(t1_im + context[origin]*t2_im, context,
            origin=origin, mode='constant', cval=0.0)
        c_eval %= context[origin]

        #Add points from 2nd threshold that were neighbours to t1-points
        t1_im[t2_im] = c_eval[t2_im] > 0

    return ret_im


def get_adaptive_threshold(im, threshold_filter=None, segments=60, 
        sigma=None, *args, **kwargs):
    """Gives a 2D surface of threshold based on smoothed local measures"""
    
    if threshold_filter is None:
        threshold_filter = filter.threshold_otsu
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


    labled, labels = get_sectioned_image(T)

    for l in range(1, labels + 1):

        if (labled==l).sum() > 1:

            T[ndimage.binary_dilation(labled == l, iterations=4)] = \
                 threshold_filter(im[labled == l], *args, **kwargs)

    return ndimage.gaussian_filter(T, sigma=sigma)

