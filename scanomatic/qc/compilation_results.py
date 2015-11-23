__author__ = 'martin'

from types import StringTypes
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import re
from itertools import izip

from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory

_img_pattern = re.compile(r".*_[0-9]{4}_[0-9.]+\.tiff$")
_time_pattern = re.compile(r'[0-9]+\.[0-9]*')


def _input_validate(f):

    def wrapped(*args, **kwargs):

        if len(args) > 0:

            if isinstance(args[0], StringTypes):

                args = list(args)
                args[0] = tuple(CompileImageAnalysisFactory.serializer.load(args[0]))

        return f(*args, **kwargs)

    return wrapped


@_input_validate
def get_grayscale_variability(project_compilation):

    data = np.array([i.fixture.grayscale.values for i in project_compilation])

    return np.var(data, axis=0) / np.mean(data, axis=0)


@_input_validate
def get_grayscale_outlier_images(project_compilation, max_distance=3.0):

    data = np.array([image.fixture.grayscale.values for image in project_compilation])
    norm = np.median(data, axis=0)
    sq_distances = np.sum((data - norm) ** 2, axis=1)
    threshold = max_distance ** 2 * np.median(sq_distances)
    return [image.image for i, image in enumerate(project_compilation) if sq_distances[i] > threshold]


@_input_validate
def plot_grayscale_histogram(project_compilation, mark_outliers=True, max_distance=3.0):

    data = np.array([image.fixture.grayscale.values for image in project_compilation])
    outliers = get_grayscale_outlier_images(project_compilation, max_distance) if mark_outliers else []
    f = plt.figure()
    f.clf()
    ax = f.gca()
    ax.imshow(data)
    ax.set_ylabel("Image index")
    ax.set_xlabel("Grayscale segment")
    ax.set_title("Grayscale segment measured values as colors" +
                 ((" (arrows, outliers)" if outliers else " (no outliers)") if mark_outliers else ""))
    if outliers:
        segments = data.shape[1]
        for outlier in outliers:
            ax.annotate(outlier.index, (segments, outlier.index), color='k')

        ax.set_xlim(0, segments)

    return f


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
def plot_positional_markers(project_compilation):

    data = np.array([(image.fixture.orientation_marks_x, image.fixture.orientation_marks_y) for
                     image in project_compilation])

    shape = np.max([image.fixture.shape for image in project_compilation], axis=0)

    f = plt.figure()
    f.clf()
    ax = f.gca()
    for x, y in data:
        ax.plot(x, y, 'x')
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_title("Position marker centers")
    return f


@_input_validate
def get_positional_markers_variability(project_compilation):

    data = _get_marker_sorted_data(project_compilation)
    return np.var(data, axis=0) / np.median(data, axis=0)


@_input_validate
def get_positional_marker_outlier_images(project_compilation, max_distance=10):

    data = _get_marker_sorted_data(project_compilation)
    norm = np.median(data, axis=0)
    sq_distances = np.sum((data - norm) ** 2, axis=(1, 2))
    irregulars = np.where(sq_distances > max_distance ** 2)[0]
    return tuple(project_compilation[i] for i in irregulars)


def _get_marker_sorted_data(project_compilation):

    data = np.array([(image.fixture.orientation_marks_x, image.fixture.orientation_marks_y) for
                     image in project_compilation])
    lengths = data.sum(axis=1)
    norm = np.median(lengths, axis=0)
    sortorder = np.argmin(np.subtract.outer(lengths, norm) ** 2, axis=-1)

    return np.array([d[:, s] for d, s in izip(data, sortorder)])