__author__ = 'martin'

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from types import StringTypes
import pandas as pd
import time
import os
import glob

from scanomatic.dataProcessing.growth_phenotypes import Phenotypes
from scanomatic.io.logger import Logger
from scanomatic.dataProcessing.phenotyper import Phenotyper
from scanomatic.io.paths import Paths

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


def load_colony_images_for_animation(analysis_directory, position, project_compilation=None):
    """

    :param analysis_directory: path to analysis directory
    :type  analysis_directory: str
    :param position: list/tuple of colony to extract. (Plate, Row, Colum)
    :type position: [int]
    :param project_compilation: Path to the associated compilation file, inferred if not submitted
    :type project_compilation: str
    :return: numpy.ndarray
    """

    analysis_directory = analysis_directory.rstrip(os.sep)
    if not project_compilation:
        experiment_directory = os.sep.join(analysis_directory.split(os.sep)[:-1])
        experiment_name = experiment_directory.split(os.sep)[-1]

        project_compilation = os.path.join(experiment_directory,
                                           Paths().project_compilation_pattern.format(experiment_name))

        if not os.path.isfile(project_compilation):

            candidates = glob.glob(os.path.join(
                experiment_directory, Paths().project_compilation_pattern.format("*")))

            if not candidates:
                _logger.error("Could not find any project.compilation file in '{0}'".format(experiment_directory))
                return np.zeros()
            elif len(candidates) != 1:
                _logger.error("Found several project.compilation files in '{0}', unsure which to use.".format(
                    experiment_directory) +
                              "Either remove one of {0} or specify compilation-file in function call".format(candidates))
                return np.zeros()

            project_compilation = candidates[0]
    else:
        experiment_directory = os.path.dirname(project_compilation)


def animate_plate_over_time(plate, initial_delay=3, delay=0.05, truncate_value_encoding=False,
                            animation_params={'action': 'run', 'index': 0, }):

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

    if 'action' not in animation_params:
        animation_params['action'] = 'run'
    if 'index' not in animation_params:
        animation_params['index'] = 0
    if 'figure' not in animation_params:
        animation_params['figure'] = plt.figure()
    if 'ax' not in animation_params:
        animation_params['ax'] = animation_params['figure'].gca()

    animation_params['ax'].cla()
    plt.ion()
    im = animation_params['ax'].imshow(plate[...,0], interpolation="nearest", vmin=vmin, vmax=vmax)

    def _animation():
        while animation_params['action'] != 'stop':
            im.set_data(plate[..., animation_params['index']])

            animation_params['index'] += 1
            animation_params['index'] %= plate.shape[-1]

            animation_params['ax'].set_title("Time {0}".format(animation_params['index']))
            animation_params['figure'].canvas.draw()
            while True:
                time.sleep(delay) if animation_params['index'] != 1 else time.sleep(initial_delay)
                if animation_params['action'] != 'pause':
                    break

    try:
        _animation()
    except KeyboardInterrupt:
        pass

    return animation_params