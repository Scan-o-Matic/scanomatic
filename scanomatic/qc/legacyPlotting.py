"""This module contains lagacy plotting methods"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#
# INTERNAL DEPENDENCIES
#

import palette
import scanomatic.logger as logger

#
# GLOBALS
#

_LOGGER = logger.Logger("Depricated visualisations")


def get_graph_styles(categories=1, n_per_cat=None, per_cat_list=None,
                     alpha=0.95):

    if per_cat_list is not None:
        per_cat_list = [range(n) for n in per_cat_list]

    if n_per_cat is not None:

        per_cat_list = [range(n_per_cat)] * categories

    if per_cat_list is None:

        return None

    #Max val = 0.25
    color_patterns = [np.array([0.25, 0.19, 0.0]),  # Orange
                      np.array([0, 0.25, 0]),  # Greens
                      np.array([0, 0, 0.25]),  # Blues
                      np.array([0, 0.25, 0.25]),  # Navy
                      np.array([0.25, 0, 0])  # Reds
                      ]

    line_styles = ['-', ':', '-.', '--']

    colors = []
    styles = []

    c_index = -1
    line_pattern = 0
    base_fraction = 0.5

    for i, cat in enumerate(per_cat_list):

        if i % len(color_patterns) == 0 and i > 0:
            line_pattern += 1
            if line_pattern > len(line_styles):
                _LOGGER.warning("Reusing styles - too many categories")
                line_pattern = 0
            c_index = 0
        else:
            c_index += 1

        for line in cat:
            color_coeff = 4 * ((1 - base_fraction) * (line + 1) /
                               float(len(cat)) + base_fraction)
            colors.append(list(color_patterns[c_index] * color_coeff) + [alpha])
            styles.append(line_styles[line_pattern])

    return styles, colors


def random_plot_on_separate_panels(xml_parser, n=10):
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    cats = []
    i = n
    while i > 5:
        cats.append(i / 5)
        i -= 5
    cats.append(i)
    styles, colors = get_graph_styles(per_cat_list=cats)
    fontP = FontProperties()
    fontP.set_size('xx-small')

    for i in xrange(4):
        ax = fig.add_subplot(2, 2, i + 1, title='Plate {0}'.format(i))
        d = xml_parser.get_plate(i)
        if d is not None:
            x, y, z, m = d.shape
            for j in xrange(n):
                tx = np.random.randint(0, x)
                ty = np.random.randint(0, y)
                ax.semilogy(d[tx, ty, :], basey=2,
                            label="{0}:{1}".format(tx, ty), color=colors[j])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                  ncol=len(cats), prop=fontP)

    return fig


def random_plot_on_same_panel(xml_parser, different_line_styles=False,
                              measurement=0):
    fig = plt.figure()
    col_maps = [np.array([0.15, 0.15, 0.10]),
                np.array([0.25, 0, 0]),
                np.array([0, 0.25, 0]),
                np.array([0, 0, 0.25])]
    if different_line_styles:
        line_styles = [':', '-.', '--', '-']
    else:
        line_styles = ['-'] * 4

    ax = fig.add_subplot(1, 1, 1, title='All on same graph')
    for i in xrange(4):
        d = xml_parser.get_plate(i)
        if d is not None:
            x, y, z, m = d.shape

            if measurement > m:
                print "Impossible, index for measurement too high"
                return None

            for j in xrange(4):
                tx = np.random.randint(0, x)
                ty = np.random.randint(0, y)

                ax.semilogy(
                    d[tx, ty, :, measurement], line_styles[i], basey=2,
                    label="{0} {1}:{2}".format(i, tx, ty),
                    color=list(col_maps[i] * (1 + 0.75 * (j + 1))) + [1])

    ax.legend(loc=4, ncol=4, title='Plate Colony_ X:Colony_Y')

    return fig


def plot(xml_parser, positionList, fig=None, measurement=0,
         phenotypes=None, ax_title=None, colorBase=plt.cm.RdBu):

    if fig is None:
        fig = plt.figure()

    fontP = FontProperties()
    fontP.set_size('xx-small')

    X = xml_parser.get_scan_times()

    ax = fig.add_subplot(1, 1, 1, title=(ax_title is None and
                                         "{0} graphs".format(len(positionList))
                                         or ax_title))

    #fullPosition = (list(p) + [measurement] for p in positionList)

    colors = palette.get(N=len(positionList), base=colorBase)

    for pos in positionList:

        Y = xml_parser[pos][..., measurement].ravel()

        c = colors.next()

        if Y is not None and np.isnan(Y).all() == False:

            try:
                ax.semilogy(X, Y, basey=2,
                            label="{0}".format(pos[:-1]), color=c)
            except:
                try:

                    ax.plot(X, Y,
                            label="Not logged! {0}".format(pos[:-1]),
                            color=c)

                except:

                    print "Failed to plot!"

        else:

            print "No data to plot!", Y

        if phenotypes is not None:

            ax.text(0.5, 0.05, str(phenotypes[0]), transform=ax.transAxes)

    if len(ax.lines) > 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.040),
                  prop=fontP)

    fig.tight_layout()
    return fig


def plot_from_list(xml_parser, position_list, fig=None, measurement=0,
                   phenotypes=None, ax_title=None):
    """
    Plots curves from an xml_parser-instance given the position_list where locations are
    described as (plate, row, column)
    """

    if fig is None:
        fig = plt.figure()

    fontP = FontProperties()
    fontP.set_size('xx-small')

    X = xml_parser.get_scan_times()

    plates = sorted([p[0] for p in position_list])
    pl = [plates[0]]
    map(lambda x: x != pl[-1] and pl.append(x), plates)

    """
    print "These are the plates {0}".format(pl)
    """

    p_pos = 1
    rows = 1
    cols = 1
    while rows * cols < len(pl):
        if cols < rows:
            cols += 1
        else:
            rows += 1

    for p in pl:

        coords = [c[1:] for c in position_list if c[0] == p]

        """
        print "Plate {0} has {1} curves to plot ({2})".format(p,
            len(coords), coords)
        """

        cats = []
        i = len(coords)
        cat_max = 5
        while i > cat_max:
            cats.append(cat_max)
            i -= cat_max
        if i > 0:
            cats.append(i)
        styles, colors = get_graph_styles(per_cat_list=cats)

        ax = fig.add_subplot(rows, cols, p_pos,
                             title=(ax_title is None and 'Plate {0}'.format(p)
                                    or ax_title))

        for j, c in enumerate(coords):

            d = xml_parser.get_colony(p, c[0], c[1])

            if d is not None and np.isnan(d).all() is False:

                try:
                    ax.semilogy(X, d[:, measurement], basey=2,
                                label="{0}".format(c), color=colors[j])
                except:
                    try:

                        ax.plot(X, d[:, measurement],
                                label="Not logged! {0}".format(c),
                                color=colors[j])

                    except:

                        pass

            else:
                print "Curve {0} is bad!".format(c)

        if phenotypes is not None:

            ax.text(0.5, 0.05, str(phenotypes[0]), transform=ax.transAxes)

        if len(ax.lines) > 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.040),
                      ncol=len(cats), prop=fontP)

        p_pos += 1

    fig.subplots_adjust(hspace=.5)
    return fig
