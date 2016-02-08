__author__ = 'martin'

from scanomatic.io.movie_writer import MovieWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from types import StringTypes
import pandas as pd

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
        horizontal_orientation=True, cm=plt.cm.RdBu_r, title_text=None, hide_axis=False, fig=None,
        save_target=None):

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

    if save_target is not None:
        fig.savefig(save_target)

    return fig


def load_phenotype_results_into_plates(file_name, phenotype_header='Generation Time'):

    data = pd.read_csv(file_name, sep='\t')
    plates = np.array([None for _ in range(data.Plate.max() + 1)], dtype=np.object)

    for plateIndex in data.Plate.unique():
        plate = data[data.Plate == plateIndex]
        if plate.any() is False:
            continue

        plates[plateIndex] = np.zeros((plate.Row.max() + 1, plate.Column.max() + 1), dtype=np.float) * np.nan

        for _, dataRow in plate.iterrows():
            plates[plateIndex][dataRow.Row, dataRow.Column] = dataRow[phenotype_header]

    return plates


def animate_plate_over_time(save_target, plate, initial_delay=3, delay=0.05, truncate_value_encoding=False,
                            animation_params={'action': 'run', 'index': 0, }, fps=3):

    masked_plate = np.ma.masked_invalid(plate).ravel()
    masked_plate = masked_plate[masked_plate.mask == False]

    if truncate_value_encoding:
        fraction = 0.1
        argorder = masked_plate.argsort()

        vmin = masked_plate[argorder[np.round(argorder.size * fraction)]]
        vmax = masked_plate[argorder[np.round(argorder.size * (1 - fraction))]]

    else:
        vmin = masked_plate.min()
        vmax = masked_plate.max()

    if 'index' not in animation_params:
        animation_params['index'] = 0
    if 'figure' not in animation_params:
        animation_params['figure'] = plt.figure()
    if 'ax' not in animation_params:
        animation_params['ax'] = animation_params['figure'].gca()
    if 'cmap' not in animation_params:
        animation_params['cmap'] = None

    animation_params['ax'].cla()
    plt.ion()
    im = animation_params['ax'].imshow(plate[..., 0], interpolation="nearest", vmin=vmin, vmax=vmax,
                                       cmap=animation_params['cmap'])

    @MovieWriter(save_target, fps=fps, fig=animation_params['figure'])
    def _animation():
        while animation_params['index'] < plate.shape[-1]:

            im.set_data(plate[..., animation_params['index']])
            animation_params['index'] += 1
            animation_params['ax'].set_title("Time {0}".format(animation_params['index']))

            yield

    _animation()

    return animation_params
