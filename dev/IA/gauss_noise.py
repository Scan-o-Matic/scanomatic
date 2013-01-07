import numpy as np

class SigmaError(Exception): pass;

def gauss_noise(im, sigma):

    if sigma <= 0:

        raise SigmaError(
            "Sigma ({0}) is not larger than 0".format(sigma))

        return None

    v = np.zeros(im.shape)
    G = im.max()

    theta = np.random.random(im.size)
    r = np.random.random(im.size)

    alpha = 2 * np.pi * theta
    beta = np.sqrt(-2 * np.log(r))

    z1 = sigma * np.cos(alpha) * beta 
    z2 = sigma * np.sin(alpha) * beta 

    flat_v = v.ravel()

    flat_v += z1
    flat_v[1:] += z2[:-1]

    f = im + v

    f[f < 0] = 0
    f[f > G] = G

    return f
