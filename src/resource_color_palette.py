import random
from itertools import permutations

try:
    from matplotlib import colors as mplColors
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


def get(N, **kwargs):
    """Generator that produces a colorseries of length N.

    Parameters:

        N   the number of colors requested

    Optional parameters:

        alpha
            If alpha is submitted and not set to None, this value
            will be added as the fourth value of the RGBA tuple,
            else RGB tuples will be generated

        base
            Either a RGB or RGBA tuple to base the colorgeneration
            on or a matplotlib color-map from which N evenly spaced
            colors will be sampled.
    """

    if ('base' in kwargs and MATPLOTLIB and
            isinstance(kwargs['base'], mplColors.Colormap)):

        for n in xrange(N):

            yield kwargs['base'](n * float(kwargs['base'].N) / N)

    else:

        if 'base' not in kwargs:

            random.seed()
            kwargs['base'] = [random.random() for x in range(3)]

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = None

        l = sum([v ** 2 for v in kwargs['base'][:3]]) ** 0.5
        cV = [v / l for v in kwargs['base'][:3]]
        maxV = float(max(cV))
        cV = [v / maxV for v in cV]
        n = 0

        while True:

            for v in permutations(cV):

                if n < N:
                    if alpha is not None:
                        yield tuple(v) + (alpha, )
                    else:
                        yield tuple(v)
                else:
                    break

                n += 1

            if n >= N:
                break

            cV = [0.5 * v for v in cV]
