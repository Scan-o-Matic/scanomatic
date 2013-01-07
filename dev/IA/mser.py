import numpy as np


def get_mser(im, tollerance=0.1, stability_threshold=10):

    S = np.arange(256)
    S_strided = np.lib.stride_tricks.as_strided(S,
        (im.shape[0], im.shape[1], S.size),
        strides=(0, 0, S.itemsize))

    im_strided = np.lib.stride_tricks.as_strided(im,
        S_strided.shape,
        strides=(im.itemsize, im.itemsize*im.shape[0], 0))
    
    Q = im_strided < S_strided

    mser = np.zeros(im.shape, dtype=np.int8)

    #Progress up along 3d axis and move stable regions over to mser
    for l in S[:-stability_threshold]:

        if Q[:,:,l].sum() > 0:

            dQ = Q[:,:,l + stability_threshold] - Q[:,:,l]
            #What to do?


    return mser
