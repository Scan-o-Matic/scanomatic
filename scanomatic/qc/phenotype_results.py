# -*- coding: utf-8 -*-
import pandas as pd
from functools import wraps
from types import StringTypes
from itertools import chain, izip, product
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib import patches as mpatches
from scipy.ndimage import label
import scipy.cluster.hierarchy as sch

from scanomatic.data_processing.growth_phenotypes import Phenotypes
from scanomatic.data_processing.phases.segmentation import CurvePhases, Thresholds, DEFAULT_THRESHOLDS, \
    get_data_needed_for_segmentation, get_curve_classification_in_steps, get_linear_non_flat_extension_per_position, \
    classifier_flat, get_barad_dur_towers
from scanomatic.data_processing.phases.features import get_phase_assignment_frequencies, CurvePhasePhenotypes, \
    get_variance_decomposition_by_phase
from scanomatic.data_processing.phenotyper import Phenotyper
from scanomatic.io.logger import Logger
from scanomatic.io.movie_writer import MovieWriter

_logger = Logger("Phenotype Results QC")

PHASE_PLOTTING_COLORS = {

    CurvePhases.Multiple: "#5f3275",
    CurvePhases.Flat: "#f9e812",

    CurvePhases.Impulse: "#ea072c",
    CurvePhases.GrowthAcceleration: "#ea5207",
    CurvePhases.GrowthRetardation: "#c797c1",

    CurvePhases.Collapse: "#0fa8aa",
    CurvePhases.CollapseAcceleration: "#0f71aa",
    CurvePhases.CollapseRetardation: "#9d0faa",

    CurvePhases.UndeterminedNonFlat: "#111111",
    CurvePhases.UndeterminedNonLinear: "#777777",
    CurvePhases.Undetermined: "#ffffff",

    "raw": "#3f040d",
    "smooth": "#849b88",
    "derivative": "#849b88",
}


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


def _setup_figure(f):

    def _wrapped(*args, **kwargs):

        def _isdefault(val):

            return val not in kwargs or kwargs[val] is None

        if _isdefault('f') and _isdefault('ax'):
            kwargs['f'] = plt.figure()
            kwargs['ax'] = kwargs['f'].gca()
        elif _isdefault('f'):
            kwargs['f'] = kwargs['ax'].figure
        elif _isdefault('ax'):
            kwargs['ax'] = kwargs['f'].gca()

        return f(*args, **kwargs)

    return _wrapped


@_validate_input
def get_position_phenotypes(phenotypes, plate, position_selection=None):

    return {phenotype.name: phenotypes.get_phenotype(phenotype)[plate][position_selection] for
            phenotype in Phenotypes}


def get_linkage_matrix(
        aligned_phases, clust_method='single', clust_metric='euclidean', nan_to_mean=True,
        drop_low_cov_phenotypes=0.01):

    if aligned_phases.ndim > 2:

        aligned_phases = aligned_phases.reshape((np.prod(aligned_phases.shape[:-1]), ) + aligned_phases.shape[-1:])

    if not np.isfinite(aligned_phases).all():
        aligned_phases = np.ma.masked_invalid(aligned_phases)

    if hasattr(aligned_phases, 'mask') and aligned_phases.mask.any():
        mask = aligned_phases.mask
        aligned_phases = aligned_phases.data
        if nan_to_mean:
            for i in range(aligned_phases.shape[1]):
                if mask[:, i].any():
                    aligned_phases[:, i][mask[:, i]] = aligned_phases[:, i][mask[:, i] == np.False_].mean()
            print("\n**WARNING** Setting masked values to phenotype means\n")
        else:
            aligned_phases[mask] = 0
            print("\n**WARNING** Setting masked values to 0\n")

    if drop_low_cov_phenotypes is not None:

        cov = aligned_phases.var(axis=0) / aligned_phases.mean(axis=0)
        aligned_phases = aligned_phases[:, cov >= drop_low_cov_phenotypes]

    print(aligned_phases.shape)
    return sch.linkage(aligned_phases, method=clust_method, metric=clust_metric)


def plot_plate_clustered_heatmap(shape, c, similarity_threshold=0.25, cmap=plt.cm.jet):

    z = sch.dendrogram(c, no_plot=True, color_threshold=similarity_threshold)

    plate = np.zeros(shape, dtype=float) * np.nan
    colors = tuple(np.unique(z["color_list"]))

    for color, ravel_pos in izip(z["color_list"], z["leaves"]):

        plate[np.unravel_index((ravel_pos, ), shape)] = colors.index(color)

    f = plt.figure("Experiment Clustering")
    ax = plt.gca()
    ax.set_title("Similarity threshold {0}, {1} clusters".format(similarity_threshold, len(colors)))
    ax.imshow(plate, cmap=cmap, interpolation='nearest')
    return f


def plot_phase_correlation_dendrogram(
        aligned_phases, phase_labels, clust_method='single', clust_metric='euclidean', cmap=plt.cm.jet, vmax=None,
        nan_to_mean=True):

    print("Reshaping")
    if aligned_phases.ndim > 2:

        aligned_phases = aligned_phases.reshape((np.prod(aligned_phases.shape[:-1]), ) + aligned_phases.shape[-1:])

    print("Building key matrix")
    aligned_phases = aligned_phases.T
    phase_key = np.ones((aligned_phases.shape[0], 3), dtype=float)
    phase_key[:, 0] = tuple(chain(*((i,) * len(v[1]) for i, v in enumerate(phase_labels))))
    phase_key[:, 1] = tuple(chain(*((v[0].value,) * len(v[1]) for i, v in enumerate(phase_labels))))
    phase_key[:, 2] = tuple(chain(*((v2.value for v2 in v[1]) for i, v in enumerate(phase_labels))))

    if vmax is None:
        vmax = phase_key.max()

    for i in range(phase_key.shape[1]):
        phase_key[:, i] *= vmax / phase_key[:, i].max()

    if hasattr(aligned_phases, 'mask') and aligned_phases.mask.any():
        mask = aligned_phases.mask
        aligned_phases = aligned_phases.data
        if nan_to_mean:
            for i in range(aligned_phases.shape[0]):
                if mask[i].any():
                    aligned_phases[i][mask[i]] = aligned_phases[i][mask[i] == np.False_].mean()
            print("\n**WARNING** Setting masked values to phenotype means\n")
        else:
            aligned_phases[mask] = 0
            print("\n**WARNING** Setting masked values to 0\n")

    f = plt.figure("Phase Correlations")
    ax_low = 0.01
    ax_height = 0.85

    dend_ax = f.add_axes([0.05, ax_low, 0.4, ax_height])
    dend_ax.axis("off")
    print("Calculating linkage")
    linkage_mat = sch.linkage(aligned_phases, method=clust_method, metric=clust_metric)
    print("Producing dendrogram")
    dendrogram = sch.dendrogram(linkage_mat, orientation='left', ax=dend_ax, no_labels=True)

    dend_ax.set_xticks([])
    dend_ax.set_yticks([])
    dend_ax.set_title("Similarity ({0}, {1}) of phases".format(clust_method, clust_metric), size='small')

    idx1 = dendrogram['leaves']

    print("Plotting heatmap")
    heat_ax = f.add_axes([0.45, ax_low, 0.15, ax_height])
    heat_ax.matshow(phase_key[idx1, :], origin='lower', aspect='auto', cmap=cmap)
    heat_ax.set_yticks([])
    heat_ax.set_xticks([0, 1, 2])
    heat_ax.set_xticklabels(["Index", "Type", "Phenotype"], rotation=45, size='small', ha='left')
    heat_ax.get_xaxis().set_tick_params(direction='out')
    heat_ax.xaxis.tick_top()
    heat_ax.spines['top'].set_visible(False)
    heat_ax.spines['left'].set_visible(False)
    heat_ax.spines['right'].set_visible(False)
    heat_ax.spines['bottom'].set_visible(False)
    print("Plotting legends")

    def proxy_line(color):
        return Line2D([0], [0], linestyle='None', marker='s', mec='k', mew=0.5, mfc=color)

    font_props = FontProperties()
    font_props.set_size('small')

    l_width = 0.34
    l_x = 0.64
    l_spacer = 0.03
    l1_height = 0.45
    l0_height = 0.09
    l2_height = 0.35

    label_space = 0.63

    loc = "lower left"

    f.legend([proxy_line(cmap(i / float(len(phase_labels)))) for i in range(len(phase_labels))],
             ["" for _ in range(len(phase_labels))],
             title="Phase Index (0 - {0})".format(len(phase_labels) - 1),
             numpoints=1, markerscale=2, ncol=len(phase_labels),
             bbox_to_anchor=(l_x, ax_low, l_width, l0_height),
             bbox_transform=f.transFigure,
             prop=font_props,
             loc=loc,
             columnspacing=0.25,
             handlelength=0.4,
             borderpad=1.05,
             mode='expand',
             )

    f.legend([proxy_line(cmap(p.value / float(len(CurvePhases)))) for p in CurvePhases],
             [re.sub(r'([a-z])([A-Z])', r'\1 \2', p.name) for p in CurvePhases],
             title="Phase Type",
             numpoints=1, markerscale=2,
             bbox_to_anchor=(l_x, ax_low + l0_height + l_spacer, l_width, l1_height),
             bbox_transform=f.transFigure,
             prop=font_props,
             loc=loc,
             labelspacing=label_space,
             mode='expand',
             )

    f.legend([proxy_line(cmap(p.value / float(len(CurvePhasePhenotypes)))) for p in CurvePhasePhenotypes],
             [re.sub(r'([a-z])([A-Z])', r'\1 \2', p.name) for p in CurvePhasePhenotypes],
             title="Phenotype",
             numpoints=1, markerscale=2,
             bbox_to_anchor=(l_x, ax_low + l0_height + l1_height + l_spacer * 2, l_width, l2_height),
             bbox_transform=f.transFigure,
             prop=font_props,
             loc=loc,
             labelspacing=label_space,
             mode='expand',
             )

    return f


@_validate_input
def plot_plate_phase_assigment_frequencies(phenotypes, plate):

    data = get_phase_assignment_frequencies(phenotypes, plate)
    data = data / data.sum(axis=1)[..., np.newaxis].astype(float)
    cum_data = np.cumsum(data, axis=1)
    f = plt.figure()

    ax = f.add_subplot(2, 1, 1)
    for idx in range(cum_data.shape[-1]):

        ax.fill_between(
            phenotypes.times,
            cum_data[:, idx],
            0 if idx == 0 else cum_data[:, idx - 1],
            color=PHASE_PLOTTING_COLORS[CurvePhases(idx)],
            alpha=0.75,
            interpolate=True)

        ax.plot(
            phenotypes.times, cum_data[:, idx],
            lw=2, label=CurvePhases(idx).name,
            color=PHASE_PLOTTING_COLORS[CurvePhases(idx)])

    tax = ax.twinx()
    a, b, c = phenotypes.smooth_growth_data[plate].shape
    var = np.ma.masked_invalid(np.log2(phenotypes.smooth_growth_data[plate]).reshape(a * b, c)).var(axis=0)

    tax.plot(phenotypes.times, var, '--', color='k', lw=2.5, label="Population size variance")

    ax.set_xlim(0, phenotypes.times.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Frequency of classification")
    ax.set_title("Plate phase classifications {0}".format(plate + 1))

    tax.set_ylabel("Log2 Population Size Variance")

    ax2 = f.add_subplot(2, 2, 3)
    ax2.set_aspect(1)

    var_decomp = (data * var[..., np.newaxis]).sum(axis=0) / var.sum()
    ax2.pie(
        var_decomp,
        colors=tuple(PHASE_PLOTTING_COLORS[CurvePhases(idx)] for idx in range(var_decomp.size)),
        labels=tuple(CurvePhases(idx).name.replace("Growth", "G ").replace("Collapse", "C")
                     if var_decomp[idx] > 0.005 else ""
                     for idx in range(var_decomp.size)))

    ax2.set_title("Total variance explained by phase type")

    ax3 = f.add_subplot(2, 2, 4)
    ax3.axis("off")
    ax3.legend(
        ax.lines + tax.lines,
        [l.get_label() for l in ax.lines + tax.lines],
        loc="center",
        fontsize='small',
        numpoints=1,
    )

    f.tight_layout()

    return f


@_validate_input
def plot_plate_phase_pop_size_variance_decomp(phenotypes, plate, relative=False, min_members=0):

    # Prepare the data
    log2_smooth = np.log2(phenotypes.smooth_growth_data[plate])
    phase_var = np.zeros((max(phase.value for phase in CurvePhases) + 1, log2_smooth.shape[-1]))
    total_var = np.zeros((log2_smooth.shape[-1],))

    for id_time in range(log2_smooth.shape[-1]):
        time_data = get_variance_decomposition_by_phase(
            log2_smooth[..., id_time], phenotypes, plate, id_time, min_members=min_members)
        for phase in time_data:
            if phase is None:
                total_var[id_time] = time_data[phase]
            elif phase.value >= 0 and np.isfinite(time_data[phase]):
                phase_var[phase.value, id_time] = time_data[phase]

    sum_phase_var = phase_var.sum(axis=0)
    phase_var = np.ma.masked_less_equal(phase_var, 0)
    phase_var.fill_value = 0
    # Plotting
    f = plt.figure()

    ax = f.add_subplot(1, 1, 1)
    cum_sum = np.zeros(sum_phase_var.shape, dtype=float)

    for idx, data in enumerate(phase_var):

        if not data.any():
            continue

        if relative:
            vals = data / sum_phase_var
            if cum_sum is not None:
                vals += cum_sum
        else:

            vals = data.copy()
            if cum_sum is not None:
                vals += cum_sum

        ax.fill_between(
            phenotypes.times,
            vals,
            cum_sum,
            where=vals.mask == np.False_,
            color=PHASE_PLOTTING_COLORS[CurvePhases(idx)],
            alpha=1,
            interpolate=True)

        ax.plot(phenotypes.times, vals, 's',
                ms=0.1,
                color=PHASE_PLOTTING_COLORS[CurvePhases(idx)],
                label=CurvePhases(idx).name)

        cum_sum[...] = [c if np.ma.is_masked(m) else m for c, m in zip(cum_sum, vals)]

    tax = ax.twinx()

    tax.plot(phenotypes.times, total_var, '--', color='k', lw=2.5, label="Total Pop-size")

    ax.set_xlim(0, phenotypes.times.max())

    if relative:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of variance by phase")

    else:
        ax.set_ylabel("Log2 Population Size Variance for curves in phase")
        ax.legend(
            ax.lines + tax.lines,
            [l.get_label() for l in ax.lines + tax.lines],
            loc="upper right",
            fontsize='x-small',
            numpoints=1,
            ncol=2,
            markerscale=40,
        )

    ax.set_xlabel("Time (h)")
    ax.set_title("Plate {0} phase variance decomposition by phase assignment".format(plate + 1))

    tax.set_ylabel("Log2 Population Size Variance")
    return f


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
def plot_all_curves_and_smoothing(phenotyper_object, id_plate, f=None,
                                  smoothing_label="Smoothed", smoothing_color=None,
                                  plot_raw=True, set_title=True, plot_from_pos=None, plot_to_pos=None):

    if f is None:
        f = plt.figure("Plate {0} curves".format(id_plate + 1))

    f.clf()

    plate = phenotyper_object.smooth_growth_data[id_plate]
    if not plot_from_pos:
        plot_from_pos = (0, 0)
    if not plot_to_pos:
        plot_to_pos = plate.shape[:2]

    plate_shape = (plot_to_pos[0] - plot_from_pos[0], plot_to_pos[1] - plot_from_pos[1])
    times = phenotyper_object.times
    if smoothing_color is None:
        smoothing_color = PHASE_PLOTTING_COLORS['smooth']

    for i, (id1, id2) in enumerate(product(range(plot_from_pos[0], plot_to_pos[0]),
                                           range(plot_from_pos[1], plot_to_pos[1]))):

        ax = f.add_subplot(plate_shape[0], plate_shape[1], i + 1)
        if set_title:
            ax.set_title("({0}, {1})".format(id1, id2))
        if plot_raw:
            curve = phenotyper_object.raw_growth_data[id_plate][id1, id2]
            ax.semilogy(times, curve, '+', basey=2, ms=2, label="Raw data", color=PHASE_PLOTTING_COLORS['raw'])
        curve = plate[id1, id2]
        ax.semilogy(times, curve, basey=2, label=smoothing_label, color=smoothing_color)

        if i == 0:
            if ax.legend_:
                ax.legend_.remove()

            ax.legend(loc="lower right", fontsize='xx-small', numpoints=1)
        else:
            ax.set_xticklabels(["" for _ in ax.get_xticklabels()])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    if set_title:
        f.tight_layout(w_pad=0.01)
    else:
        f.tight_layout(h_pad=0.01, w_pad=0.01)

    return f


@_validate_input
def plot_phases_all_curves(
        phenotyper_object, id_plate, f=None, set_title=False, plot_from_pos=None, plot_to_pos=None,
        plot_derivative=False, hide_all_legends=True):

    if f is None:
        f = plt.figure("Plate {0} phases".format(id_plate + 1))
    else:
        f.clf()

    plate = phenotyper_object.smooth_growth_data[id_plate]
    if not plot_from_pos:
        plot_from_pos = (0, 0)
    if not plot_to_pos:
        plot_to_pos = plate.shape[:2]

    plate_shape = (plot_to_pos[0] - plot_from_pos[0], plot_to_pos[1] - plot_from_pos[1])
    times = phenotyper_object.times

    for i, (id1, id2) in enumerate(product(range(plot_from_pos[0], plot_to_pos[0]),
                                           range(plot_from_pos[1], plot_to_pos[1]))):

        ax = f.add_subplot(plate_shape[0], plate_shape[1], i + 1)

        if set_title:
            ax.set_title("({0}, {1})".format(id1, id2))

        print("Constructing plot {0}".format((id1, id2)))

        curve = plate[id1, id2]
        deriv = phenotyper_object.get_derivative(id_plate, (id1, id2)) if plot_derivative else None
        phases = phenotyper_object.get_curve_phases(id_plate, id1, id2)

        plot_phases_from_data(times, np.log2(curve), phases, ax=ax, f=f, colors=PHASE_PLOTTING_COLORS, deriv=deriv,
                              plot_legend=False, set_labels=i == 0 and not hide_all_legends, layout=False)

    if set_title:
        f.tight_layout(w_pad=0.01)
    else:
        f.tight_layout(h_pad=0.01, w_pad=0.01)

    return f


@_validate_input
def plot_phases_legend(f=None, colors=None):

    if f is None:
        f = plt.figure("Phases legend")
    else:
        f.clf()

    if colors is None:
        colors = PHASE_PLOTTING_COLORS

    ax = f.gca()

    legend = []
    for phase in CurvePhases:
        if phase is CurvePhases.Multiple:
            continue

        legend.append(mpatches.Patch(color=colors[phase], label=re.sub(r"([a-z])([A-Z])", r'\1 \2', phase.name)))

    ax.legend(handles=legend, loc='center', fontsize='large', markerscale=0.66, ncol=2,
              bbox_to_anchor=(0., 0., 1., 1.), borderaxespad=0)

    ax.axis("off")

    return f


@_validate_input
@_setup_figure
def plot_barad_dur_plot(phenotyper_object, plate, pos, plot_curve=False, plot_final_phases=False, f=None, ax=None):

    model = get_data_needed_for_segmentation(phenotyper_object, plate, pos, DEFAULT_THRESHOLDS)
    ext_val, _ = get_linear_non_flat_extension_per_position(model, DEFAULT_THRESHOLDS)
    model.phases[classifier_flat(model)[1]] = CurvePhases.Flat.value

    flat_filt = model.phases == CurvePhases.Flat.value
    """:type: numpy.ndarray"""

    flat_parts, n_flats = label(flat_filt)

    non_flat_filt = model.phases != CurvePhases.Flat.value
    """:type: numpy.ndarray"""

    non_flat_parts, n_nonflats = label(non_flat_filt)
    towers, n_towers = get_barad_dur_towers(ext_val, non_flat_filt, DEFAULT_THRESHOLDS)

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Length of local linearity in number of indices")
    ax.set_title(u"Barad-dûr plot of plate {0} pos {1}".format(model.plate, model.pos))

    legend_handles = []
    legend_labels = []

    line_out = None
    for i in range(1, n_flats + 1):
        line_out = ax.plot(
            phenotyper_object.times[flat_parts == i], ext_val[flat_parts == i], ':', color='r',
            label="Flat segment")[0]

    legend_handles.append(line_out)
    legend_labels.append(line_out.get_label() if line_out else "Flat segment (missing)")

    line_in = None
    for i in range(1, n_nonflats + 1):
        line_in = ax.plot(
            phenotyper_object.times[non_flat_parts == i], ext_val[non_flat_parts == i], '-', color='k',
            label="Non-flat segment")[0]

    legend_handles.append(line_in)
    legend_labels.append(line_in.get_label() if line_in else "Non-flat segment (missing)")

    if plot_final_phases:

        non_flat_linear = (phenotyper_object.get_curve_phases(
            model.plate, model.pos[0], model.pos[1]) == 4).filled(False)

        phases, n_phases = label(non_flat_linear)
        line_phases = None
        for i in range(1, n_phases + 1):
            line_phases = ax.fill_between(
                model.times[phases == i],
                ext_val[phases == i],
                0,
                color='r', edgecolor='r', alpha=0.9, lw=0,
                label='Linear Phase(s)')

        legend_handles.append(line_phases)
        legend_labels.append(line_phases.get_label() if line_phases else "Linear Phase(s) (missing)")

    tower_line = None
    if n_towers:
        # ground = ext_val[non_flat_filt].min()

        for i in range(1, n_towers + 1):
            tower_line = ax.fill_between(phenotyper_object.times[towers == i],
                                         ext_val[towers == i],
                                         0, color='k', edgecolor='k', alpha=0.9, lw=0,
                                         label='Tower(s)')

    legend_handles.append(tower_line)
    legend_labels.append(tower_line.get_label() if tower_line else "Tower(s) (missing)")

    if plot_curve:
        tax = ax.twinx()
        line_curve = tax.plot(model.times, model.log2_curve, 'gray', lw=2, label='Growth curve')[0]
        tax.set_ylabel("Population Size [Cells, log2]")
        legend_handles.append(line_curve)
        legend_labels.append(line_curve.get_label())
        tax.spines['top'].set_visible(False)
        tax.spines['left'].set_visible(False)

    ax.legend(legend_handles, legend_labels, loc='lower right')

    ax.set_xlim((0, model.times.max()))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return f


@_validate_input
@_setup_figure
def plot_curve_and_derivatives(phenotyper_object, plate, pos, thresholds=DEFAULT_THRESHOLDS, show_thresholds=True,
                               ax=None, f=None, **kwargs):
    """Plots a log2_curve and both its derivatives as calculated and
    smoothed by scanomatic during the phase sectioning.

    Args:
        phenotyper_object: A Phenotyper instance
        plate: the plate index
        pos: the position tuple on the plate
        thresholds: A thresholds dictionary
        show_thresholds: If include horizontal lines for applicable thresholds
        ax: Figure axes to plot in
        f: Figure (used if not ax supplied)

    Returns: A matplotlib figure

    """

    model = get_data_needed_for_segmentation(phenotyper_object, plate, pos, thresholds)

    return plot_curve_and_derivatives_from_model(model, thresholds, show_thresholds, ax=ax, **kwargs)


@_setup_figure
def plot_curve_and_derivatives_from_model(model, thresholds=DEFAULT_THRESHOLDS, show_thresholds=False, f=None, ax=None,
                                          fill_alpha=0.35):
    thresholds_width = 1.5

    times = model.times
    nan_arr = np.ones_like(model.d2yd2t) * np.nan
    for val, color in izip((-1, 0, 1), ('r', 'k', 'g')):
        filt = model.d2yd2t_signs != val
        ax.plot(times, np.choose(filt, (model.d2yd2t, nan_arr)), lw=2.5, color=color)

    ax.set_ylabel("d2y/dt2 ", color='g')
    ax.fill_between(times, model.d2yd2t, 0, color='g', alpha=fill_alpha)

    if show_thresholds:

        t1 = model.d2yd2t.std() * thresholds[Thresholds.SecondDerivativeSigmaAsNotZero]
        ax.axhline(-t1, linestyle='--', color='g', lw=thresholds_width)
        ax.axhline(t1, linestyle='--', color='g', lw=thresholds_width)

    ax2 = ax.twinx()
    nan_arr = np.ones_like(model.d2yd2t) * np.nan
    for val, color in izip((-1, 0, 1), ('r', 'k', 'g')):
        filt = model.dydt_signs != val
        ax2.plot(times, np.choose(filt, (model.dydt, nan_arr)), lw=2.5, color=color)

    ax2.fill_between(times, model.dydt, 0, color='r', alpha=fill_alpha)
    ax2.set_ylabel("dy/dt", color='r')

    if show_thresholds:

        t1 = thresholds[Thresholds.FlatlineSlopRequirement]

        ax2.axhline(t1, linestyle='--', color='r', lw=thresholds_width)
        ax2.axhline(-t1, linestyle='--', color='r', lw=thresholds_width)

    ax3 = ax.twinx()
    ax3.plot(times, model.log2_curve, color='k', lw=2)
    ax3.yaxis.labelpad = -40
    ax3.set_ylabel("Log2 cells")
    ax3.set_yticks(ax3.get_yticks()[1:-1])

    for tick in ax3.yaxis.get_major_ticks():
        tick.set_pad(-5)
        tick.label2.set_horizontalalignment('right')

    ax.set_xlabel("Hours")
    ax.set_title("Curve {0} and its derviatives".format((model.plate, model.pos)))
    """
    legend = ax.legend(
        [ax3.lines[0], ax2.lines[0], ax.lines[0]],
        ['growth', 'dy/dt', 'ddy/dt'],
        loc='lower right',
        bbox_to_anchor=(0.9, 0))

    legend.get_frame().set_facecolor('white')
    """
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


@_setup_figure
def plot_phases(phenotypes, plate, position, segment_alpha=1, f=None, ax=None, colors=None, save_target=None,
                loc="lower right", plot_deriv=False):

    if not isinstance(phenotypes, Phenotyper):
        phenotypes = Phenotyper.LoadFromState(phenotypes)

    model = get_data_needed_for_segmentation(phenotypes, plate, position, DEFAULT_THRESHOLDS)
    model.phases = phenotypes.get_curve_phases(plate, position[0], position[1])

    plot_phases_from_model(model, ax=ax, colors=colors, segment_alpha=segment_alpha, loc=loc,
                           plot_deriv=plot_deriv)

    ax.set_title("Curve phases for plate {0}, position ({1}, {2})".format(plate, *position))

    if save_target:
        f.savefig(save_target)

    return f


@_setup_figure
def plot_phases_from_model(model, ax=None, f=None, colors=None, segment_alpha=1, loc="lower right",
                           plot_deriv=False):

    times = model.times
    phases = model.phases
    log2_curve = model.log2_curve
    return plot_phases_from_data(times, log2_curve, phases, ax=ax, f=f, colors=colors, segment_alpha=segment_alpha,
                                 loc=loc, deriv=model.dydt if plot_deriv else None)


@_setup_figure
def plot_phases_from_data(times, log2_curve, phases, ax=None, f=None, colors=None, segment_alpha=1, loc="lower right",
                          deriv=None, plot_legend=True, set_labels=True, layout=True):

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
            right = np.linspace(times[min(log2_curve.size - 1, right + 1)], times[right], 3)[1]
            span = ax.axvspan(left, right, color=colors[phase], alpha=segment_alpha, label=phase.name)
            if phase not in legend:
                legend[phase] = span

    ax.plot(times, log2_curve, "-", color=colors["smooth"], lw=2)
    if deriv is not None:

        if times.size > deriv.size:
            delta = times.size - deriv.size
            assert delta % 2 == 0, "The derivative to times offset is not multiple of 2"
            delta /= 2
            deriv_times = times[delta:-delta]
        else:
            deriv_times = times

        tax = ax.twinx()
        tax.plot(deriv_times, deriv, "--", color=colors["derivative"], lw=2)
        if set_labels:
                tax.set_ylabel("dY/dt used for phases")
        else:
            tax.set_yticks([])

    ax.set_xlim(xmin=times[0], xmax=times[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if set_labels:
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Pop Size [Cells, log2]")
    else:
        ax.set_xticklabels(["" for _ in ax.get_xticklabels()])
        ax.set_yticklabels(["" for _ in ax.get_yticklabels()])

    if plot_legend and legend:
        ax.legend(loc=loc, handles=list(legend.values()))

    if layout:
        f.subplots_adjust(hspace=0.3, wspace=0.15, top=0.95)

    return f


def plot_phase_segmentation_in_steps(phenotyper, plate, position, plot_deriv=True, **kwargs):

    steps, model = get_curve_classification_in_steps(phenotyper, plate, position)
    plots = len(steps)

    if 'colors' in kwargs:
        colors = kwargs['colors']
    else:
        colors = PHASE_PLOTTING_COLORS

    if 'f' in kwargs:
        f = kwargs['f']
        del kwargs['f']
        f.clf()
    else:
        f = plt.figure("curve_{0}_{1}_{2}_segmentation_steps".format(plate, *position))

    legend_space = 2
    cols = int(np.ceil(np.sqrt(plots + legend_space)))
    rows = int(np.ceil(float(plots + legend_space) / cols))

    for i in range(plots):
        ax = f.add_subplot(rows, cols, i + 1)
        ax.set_title("Step {0}".format(i))
        plot_phases_from_data(model.times, model.log2_curve, steps[i], ax=ax, f=f,
                              deriv=model.dydt if plot_deriv else None,
                              plot_legend=False, set_labels=i==0,
                              **kwargs)

    ax = f.add_subplot(rows, cols, plots + 1)
    ax.axis("off")

    legend = []
    for phase in CurvePhases:
        if phase is CurvePhases.Multiple:
            continue

        legend.append(mpatches.Patch(color=colors[phase], label=re.sub(r'([a-z])([A-Z])', r'\1 \2', phase.name)))

    plt.legend(handles=legend, loc='center', fontsize='small',
               markerscale=0.66, ncol=2, bbox_to_anchor=(0., 0., 2., 1.), borderaxespad=0)

    f.tight_layout(h_pad=0.01, w_pad=0.01)

    return f
