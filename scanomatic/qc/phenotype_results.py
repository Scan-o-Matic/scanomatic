from scipy.ndimage import label
from scipy import signal

from scanomatic.data_processing.curve_phase_phenotypes import CurvePhases

from scanomatic.io.movie_writer import MovieWriter
import matplotlib

import numpy as np
from types import StringTypes
import pandas as pd
from functools import wraps

from scanomatic.data_processing.growth_phenotypes import Phenotypes
from scanomatic.io.logger import Logger
from scanomatic.data_processing.phenotyper import Phenotyper

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

_logger = Logger("Phenotype Results QC")

PHASE_PLOTTING_COLORS = {
    CurvePhases.Multiple: "#5f3275",
    CurvePhases.Flat: "#f9e812",
    CurvePhases.GrowthAcceleration: "#ea5207",
    CurvePhases.Impulse: "#ea072c",
    CurvePhases.GrowthRetardation: "#c797c1",
    CurvePhases.Collapse: "#0fa8aa",
    CurvePhases.CollapseAcceleration: "#0f71aa",
    CurvePhases.CollapseRetardation: "#9d0faa",
    "raw": "#3f040d",
    "smooth": "#849b88"}


@wraps
def _validate_input(f):

    def _wrapped(*args, **kwargs):

        if len(args) > 0 and isinstance(args[0], StringTypes):

            args = list(args)
            args[0] = Phenotyper.LoadFromState(args[0])
        elif 'phenotypes' in kwargs and isinstance(kwargs['phenotypes'], StringTypes):
            kwargs['phenotypes'] = Phenotyper.LoadFromState(kwargs['phenotypes'])

        return f(*args, **kwargs)

    return _wrapped


@_validate_input
def get_position_phenotypes(phenotypes, plate, position_selection=None):

    return {phenotype.name: phenotypes.get_phenotype(phenotype)[plate][position_selection] for phenotype in Phenotypes}


@_validate_input
def plot_plate_heatmap(
        phenotypes, plate_index, measure=None, use_common_value_axis=True, vmin=None, vmax=None, show_color_bar=True,
        horizontal_orientation=True, cm=plt.cm.RdBu_r, title_text=None, hide_axis=False, fig=None,
        save_target=None, normalized=False):

    if measure is None:
        measure = Phenotypes.GenerationTime

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
        plate_data = phenotypes.get_phenotype(measure, normalized=normalized)[plate_index].astype(np.float)
    except ValueError:
        _logger.error("The phenotype {0} is not scalar and thus can't be displayed as a heatmap".format(measure))
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
            for plate in phenotypes.get_phenotype(measure):
                vmin = min(vmin, plate.min())
                vmax = max(vmax, plate.max())

    font = {'family': 'sans',
            'weight': 'normal',
            'size': 6}

    matplotlib.rc('font', **font)

    ax.imshow(
        plate_data.mask,
        vmin=0,
        vmax=2,
        interpolation="nearest",
        cmap=plt.cm.Greys)

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


@_validate_input
def plot_curve_and_derivatives(phenotyper_object, plate, pos):
    """Plots a curve and both its derivatives as calculated and
    smoothed by scanomatic during the phase sectioning.

    Args:
        phenotyper_object: A Phenotyper instance
        plate: the plate index
        pos: the position tuple on the plate

    Returns: A matplotlib figure

    """
    curve = np.log2(phenotyper_object.smooth_growth_data[plate][pos])
    times = phenotyper_object.times
    dydt = signal.convolve(
        phenotyper_object.get_derivative(plate, pos),
        [0.1, 0.25, 0.3, 0.25, 0.1], mode='valid')
    d_offset = (phenotyper_object.times.size - dydt.size) / 2

    dydt = np.hstack(
        ([dydt[0] for _ in range(d_offset)],
         dydt,
         [dydt[-1] for _ in range(d_offset)]))

    ddydt = signal.convolve(dydt, [1, 1, 1, 0, -1, -1, -1], mode='valid')
    dd_offset = (phenotyper_object.times.size - ddydt.size) / 2
    ddydt = np.hstack(
        ([ddydt[0] for _ in range(dd_offset)],
         ddydt,
         [ddydt[-1] for _ in range(dd_offset)]))

    f = plt.figure()
    ax = f.gca()
    ax.plot(times, ddydt, color='g')
    ax.set_ylabel("ddy/dt")
    ax.fill_between(times, ddydt, 0, color='g', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(times, dydt, color='r')
    ax2.fill_between(times, dydt, 0, color='r', alpha=0.7)
    ax2.set_ylabel("dy/dt")
    ax3 = ax.twinx()
    ax3.plot(times, curve, color='k', lw=2)
    ax3.yaxis.labelpad = -40
    ax3.set_ylabel("Log2 cells")

    for tick in ax3.yaxis.get_major_ticks():
        tick.set_pad(-5)
        tick.label2.set_horizontalalignment('right')

    ax.set_xlabel("Hours")
    ax.set_title("Curve {0} and its derviatives".format((plate, pos)))
    ax.legend(
        ax3.lines + ax2.lines + ax.lines,
        ['growth', 'dy/dt', 'ddy/dt'],
        loc='lower right',
        bbox_to_anchor=(0.9, 0))
    f.tight_layout()
    return f

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


def animate_plate_over_time(save_target, plate, truncate_value_encoding=False, index=None, fig=None, ax=None, fps=3,
                            cmap=None):

    if index is None:
        index = 0

    masked_plate = np.ma.masked_invalid(plate).ravel()
    masked_plate = masked_plate[masked_plate.mask == np.False_]

    if truncate_value_encoding:
        fraction = 0.1
        argorder = masked_plate.argsort()

        vmin = masked_plate[argorder[np.round(argorder.size * fraction)]]
        vmax = masked_plate[argorder[np.round(argorder.size * (1 - fraction))]]

    else:
        vmin = masked_plate.min()
        vmax = masked_plate.max()

    @MovieWriter(save_target, fps=fps, fig=fig)
    def _animation():

        if ax is None:
            ax = fig.gca()

        im = ax.imshow(plate[..., 0], interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)

        while index < plate.shape[-1]:

            im.set_data(plate[..., index])
            ax.set_title("Time {0}".format(index))
            index += 1

            yield

    return _animation()


def plot_segments(save_target, phenotypes, position, segment_alpha=0.3, f=None, colors=None):

    if not isinstance(phenotypes, Phenotyper):
        phenotypes = Phenotyper.LoadFromState(phenotypes)

    times = phenotypes.times
    curve_smooth = phenotypes.smooth_growth_data[position[0]][position[1], position[2]]
    curve_raw = phenotypes.raw_growth_data[position[0]][position[1], position[2]]
    phases = phenotypes.get_curve_segments(*position)

    if f is None:
        f = plt.figure()

    ax = f.gca()

    plot_phases(ax, curve_raw, curve_smooth, times, phases, colors=colors, segment_alpha=segment_alpha)

    if save_target:
        f.savefig(save_target)

    return f


def plot_phases(ax, curve_raw, curve_smooth, times, phases, colors=None, segment_alpha=0.3):

    if colors is None:
        colors = PHASE_PLOTTING_COLORS

    legend = {}

    # noinspection PyTypeChecker
    for phase in CurvePhases:

        if phase == CurvePhases.Undetermined:
            continue

        labels, label_count = label(phases == phase.value)
        for id_label in range(1, label_count + 1):
            positions = np.where(labels == id_label)[0]
            left = positions[0]
            right = positions[-1]
            left = np.linspace(times[max(left - 1, 0)], times[left], 3)[1]
            right = np.linspace(times[min(curve_raw.size - 1, right + 1)], times[right], 3)[1]
            span = ax.axvspan(left, right, color=colors[phase], alpha=segment_alpha, label=phase.name)
            if phase not in legend:
                legend[phase] = span

    ax.semilogy(times, curve_raw, "+", basey=2, color=colors["raw"], ms=6)
    ax.semilogy(times, curve_smooth, "-", basey=2, color=colors["smooth"], lw=2)

    ax.set_xlim(xmin=times[0], xmax=times[-1])
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Population Size [cells]")

    ax.legend(loc="lower right", handles=list(legend.values()))
