
import numpy as np
from scipy import ndimage as scind

def get_denoise_segments(im, **kwargs):

    erode_im = scind.binary_erosion(im, **kwargs)
    reconstruct_im = scind.binary_propagation(erode_im, mask=im)
    tmp = np.logical_not(reconstruct_im)
    erode_tmp = scind.binary_erosion(tmp, **kwargs)
    reconstruct_final = np.logical_not(scind.binary_propagation(
        erode_tmp, mask=tmp))

    return reconstruct_final

def get_segments_by_size(im, min_size, max_size=-1, inplace=True):

    if inplace:
        out = im
    else:
        out = im.copy()

    if max_size == -1:
        max_size = im.size

    labled_im, labels = scind.label(im)
    sizes = scind.sum(im, labled_im, range(labels + 1))

    mask_sizes = np.logical_or(sizes < min_size, sizes > max_size)
    remove_pixels = mask_sizes[labled_im]

    out[remove_pixels] = 0

    return out

def demo_segments_by_size(im, box_size=(105, 105)):

    from matplotlib import pyplot as plt

    im_filtered = get_segments_by_size(im, min_size=100,
        max_size=box_size[0]*box_size[1], inplace=False)
    labled, labels = scind.label(im_filtered)
    centra = scind.center_of_mass(im_filtered, labled, range(1, labels+1))
    X, Y = np.array(centra).T

    plt.imshow(im_filtered)
    plt.plot(Y, X, 'g+', ms=10, mew=2)
    plt.ylim(0, im_filtered.shape[0])
    plt.xlim(0, im_filtered.shape[1])
