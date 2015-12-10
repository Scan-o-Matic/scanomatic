__author__ = 'martin'

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from types import StringTypes
from itertools import izip

from scanomatic.dataProcessing.growth_phenotypes import Phenotypes
from scanomatic.io.logger import Logger
from scanomatic.dataProcessing.phenotyper import Phenotyper

_logger = Logger("Phenotype Results QC")

def _validate_input(f):

    def wrapped(*args, **kwargs):

        if len(args) > 0 and isinstance(args[0], StringTypes):

            args = list(args)
            args[0] = Phenotyper.LoadFromState(args[0]).phenotypes
        elif 'phenotypes' in kwargs and isinstance(kwargs['phenotypes'], StringTypes):
            kwargs['phenotypes'] = Phenotyper.LoadFromState(kwargs['phenotypes']).phenotypes

        return f(*args, **kwargs)

    return wrapped


@_validate_input
def get_position_phenotypes(phenotypes, plate, position_selection=None):

    return {phenotype.name: phenotypes[plate][position_selection][phenotype.value] for phenotype in Phenotypes}


@_validate_input
def plot_plate_heatmap(
        phenotypes, plate_index, measure=None, use_common_value_axis=True, vmin=None, vmax=None, show_color_bar=True,
        horizontal_orientation=True, cm=plt.cm.RdBu_r, title_text=None, hide_axis=False, fig=None, show_figure=True):

    if measure is None:
        measure = Phenotypes.GenerationTime.value
    elif isinstance(measure, Phenotypes):
        measure = measure.value

    if fig is None:
        fig = plt.figure()

    cax = None

    if len(fig.axes):
        ax = fig.axes[0]
        if len(fig.axes) == 2:
            cax = fig.axes[1]
            cax.cla()
            fig.delaxes(cax)
            cax = None
        ax.cla()
    else:
        ax = fig.gca()

    if title_text is not None:
        ax.set_title(title_text)

    try:
        plate_data = phenotypes[plate_index][..., measure].astype(np.float)
    except ValueError:
        _logger.error("The phenotype {0} is not scalar and thus can't be displayed as a heatmap".format(
            Phenotypes(measure)))
        return fig

    if not horizontal_orientation:
        plate_data = plate_data.T

    if plate_data[np.isfinite(plate_data)].size == 0:
        _logger.error("No finite data")
        return fig

    if None in (vmin, vmax):
        vmin = plate_data[np.isfinite(plate_data)].min()
        vmax = plate_data[np.isfinite(plate_data)].max()

        if use_common_value_axis:
            for plate in phenotypes:

                plate = np.ma.masked_invalid(plate[..., measure].astype(np.float))
                vmin = min(vmin, plate.min())
                vmax = max(vmax, plate.max())

    font = {'family': 'sans',
            'weight': 'normal',
            'size': 6}

    matplotlib.rc('font', **font)

    im = ax.imshow(
        plate_data,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        cmap=cm)

    if show_color_bar:
        divider = make_axes_locatable(ax)
        if cax is None:
            cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax)

    if hide_axis:
        ax.set_axis_off()

    fig.tight_layout()
    if show_figure:
        fig.show()

    return fig

