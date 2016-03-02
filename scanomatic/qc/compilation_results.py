from scanomatic.io.movie_writer import MovieWriter

from types import StringTypes
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import re
from itertools import izip
import time

from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory

_img_pattern = re.compile(r".*_[0-9]{4}_[0-9.]+\.tiff$")
_time_pattern = re.compile(r'[0-9]+\.[0-9]*')


def _input_validate(f):

    def wrapped(*args, **kwargs):

        if len(args) > 0:

            if isinstance(args[0], StringTypes):

                args = list(args)
                args[0] = CompileImageAnalysisFactory.serializer.load(args[0])

        return f(*args, **kwargs)

    return wrapped


def simulate_positioning(project_compilation, positioning):

    assert positioning in ('detected', 'probable', 'one-time'), "Not understood positioning mode"

    positions = np.array([(image.fixture.orientation_marks_x, image.fixture.orientation_marks_y)
                          for image in project_compilation])

    if positioning == "probable":

        positions[:] = np.round(np.median(positions, axis=0))

    elif positioning == "one-time":

        positions[:] = positions[-1]

    return positions


@_input_validate
def get_grayscale_variability(project_compilation):

    data = np.array([i.fixture.grayscale.values for i in project_compilation])

    return np.var(data, axis=0) / np.mean(data, axis=0)


@_input_validate
def get_grayscale_outlier_images(project_compilation, max_distance=3.0, only_image_indices=False):

    data = np.array([image.fixture.grayscale.values for image in project_compilation])
    norm = np.median(data, axis=0)
    sq_distances = np.sum((data - norm) ** 2, axis=1)
    threshold = max_distance ** 2 * np.median(sq_distances)
    return [(i if only_image_indices else image) for i, image in enumerate(project_compilation)
            if sq_distances[i] > threshold]


@_input_validate
def plot_grayscale_histogram(project_compilation, mark_outliers=True, max_distance=3.0, save_target=None):

    data = [image.fixture.grayscale.values for image in project_compilation]
    length = max(len(v) for v in data if v is not None)
    empty = np.zeros((length,), dtype=float) * np.inf
    data = [empty if d is None else d for d in data]
    data = np.array(data)
    if mark_outliers:
        outliers = get_grayscale_outlier_images(project_compilation, max_distance) if mark_outliers else []
    else:
        outliers = None
    f = plt.figure()
    f.clf()
    ax = f.gca()
    ax.imshow(data, interpolation='nearest', aspect='auto')
    ax.set_ylabel("Image index")
    ax.set_xlabel("Grayscale segment")
    ax.set_title("Grayscale segment measured values as colors" +
                 ((" (arrows, outliers)" if outliers else " (no outliers)") if mark_outliers else ""))
    if outliers:
        segments = data.shape[1]
        for outlier in outliers:
            ax.annotate(outlier.image.index, (segments, outlier.image.index), color='k')

        ax.set_xlim(0, segments)

    if save_target is not None:
        f.savefig(save_target)

    return f


@_input_validate
def animate_marker_positions(project_compilation, fig=None, slice_size=201,
                             positioning='detected', save_target="marker_positions.avi",
                             title="Position markers", comment="", fps=12):

    assert slice_size % 2 == 1, "Slice size may not be even"

    positions = simulate_positioning(project_compilation, positioning)

    paths = [image.image.path if os.path.isfile(image.image.path) else os.path.basename(image.image.path)
             for image in project_compilation]

    plt.ion()
    if fig is None:
        fig = plt.figure()

    fig.clf()

    images = [None for _ in range(positions.shape[-1])]
    half_slice_size = np.floor(slice_size / 2.0)

    for idx in range(len(images)):
        ax = fig.add_subplot(len(images), 1, idx + 1)
        images[idx] = ax.imshow(
            np.zeros((slice_size, slice_size), dtype=np.float), cmap=plt.cm.gray, vmin=0, vmax=255)
        ax.axvline(half_slice_size, color='c')
        ax.axhline(half_slice_size, color='c')

    def make_cutout(img, pos_y, pos_x):
        cutout = np.zeros((slice_size, slice_size), dtype=np.float) * np.nan
        cutout[abs(min(pos_x - half_slice_size, 0)): min(cutout.shape[0], img.shape[0] - pos_x),
               abs(min(pos_y - half_slice_size, 0)): min(cutout.shape[1], img.shape[1] - pos_y)] = \
            img[max(pos_x - half_slice_size, 0): min(pos_x + half_slice_size + 1, img.shape[0]),
                max(pos_y - half_slice_size, 0): min(pos_y + half_slice_size + 1, img.shape[1])]
        return cutout

    @MovieWriter(save_target, title=title, comment=comment, fps=fps, fig=fig)
    def _animate():


        data = [None for _ in range(positions.shape[0])]

        for index in range(positions.shape[0]):

            if data[index] is None:

                image = plt.imread(paths[index])
                data[index] = []
                for im_index, im in enumerate(images):
                    im_slice = make_cutout(image, *positions[index, :, im_index])
                    im.set_data(im_slice)
                    data[index].append(im_slice)
            else:
                for im_index, im in enumerate(images):
                    im.set_data(data[index][im_index])

            fig.axes[0].set_title("Time {0}".format(index))
            yield

    _animate()

    return fig


@_input_validate
def get_irregular_intervals(project_compilation, max_deviation=0.05):

    return _get_irregular_intervals([i.image.time_stamp for i in project_compilation], max_deviation)


def get_irregular_intervals_from_file_names(directory, max_deviation=0.05):

    images = [float(_time_pattern.findall(f)[-1]) for f in sorted(glob.glob(os.path.join(directory, "*.tiff")))
              if _img_pattern.match(f)]

    return _get_irregular_intervals(images, max_deviation)


def _get_irregular_intervals(data, max_deviation):

    diff = np.diff(data)
    norm = np.abs(np.median(diff))
    irregulars = np.where(np.abs(1 - diff / norm) > max_deviation)[0]
    return tuple((i + 1, diff[i]) for i in irregulars)


@_input_validate
def plot_positional_markers(project_compilation, save_target=None):

    data = _get_marker_sorted_data(project_compilation)

    shape = np.max([image.fixture.shape for image in project_compilation], axis=0)

    scans = data.shape[0]

    f = plt.figure()
    f.clf()
    ax = f.add_subplot(2, 2, 1)
    for x, y in data:
        ax.plot(x, y, 'x')
    for i in range(data.shape[2])[:3]:
        ax.annotate(i + 1, (data[:, 0, i].mean(), data[:, 1, i].mean()), textcoords='offset points', xytext=(10, -5))
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set(adjustable='box', aspect=1)
    ax.set_title("Position marker centers")
    cm = plt.get_cmap("Blues")

    for i in range(data.shape[2])[:3]:

        ax = f.add_subplot(2, 2, i + 2)
        x = data[:, 0, i]
        y = data[:, 1, i]
        x -= x.min()
        y -= y.min()
        im = np.zeros((round(x.max()) + 1, round(y.max()) + 1), dtype=int)
        for x_val, y_val in izip(x, y):
            im[round(x_val), round(y_val)] += 1

        print "Marker ", i + 1, '\n', im, '\n'
        ax.imshow(im.T, interpolation="none", cmap=cm, vmin=0, vmax=scans)
        for idx0, series in enumerate(im):
            for idx1, value in enumerate(series):
                ax.annotate(value, (idx0, idx1), color=cm(0.9) if value < scans/2.0 else cm(0.1), ha='center', va='center')
        ax.set_title("Marker {0} pos freqs".format(i + 1))
        ax.axis('off')

    f.tight_layout()

    if save_target is not None:
        f.savefig(save_target)

    return f


@_input_validate
def get_positional_markers_variability(project_compilation):

    data = _get_marker_sorted_data(project_compilation)
    return np.var(data, axis=0) / np.median(data, axis=0)


@_input_validate
def get_positional_marker_outlier_images(project_compilation, max_distance=4, only_image_indices=False):

    data = _get_marker_sorted_data(project_compilation)
    norm = np.median(data, axis=0)
    sq_distances = np.sum((data - norm) ** 2, axis=(1, 2))
    irregulars = np.where(sq_distances > max_distance ** 2)[0]
    return irregulars if only_image_indices else tuple(project_compilation[i] for i in irregulars)


def _get_marker_sorted_data(project_compilation):

    data = np.array([(image.fixture.orientation_marks_x, image.fixture.orientation_marks_y) for
                     image in project_compilation])
    lengths = data.sum(axis=1)
    norm = np.median(lengths, axis=0)
    sortorder = np.argmin(np.subtract.outer(lengths, norm) ** 2, axis=-1)

    return np.array([d[:, s] for d, s in izip(data, sortorder)])


@_input_validate
def get_images_with_irregularities(project_compilation, only_image_indices=False):

    data = set(get_grayscale_outlier_images(project_compilation, only_image_indices=only_image_indices)).union(
        get_positional_marker_outlier_images(project_compilation, only_image_indices=only_image_indices))

    if only_image_indices:
        return sorted(data)
    else:
        return sorted(data, key=lambda x: x.image.index)