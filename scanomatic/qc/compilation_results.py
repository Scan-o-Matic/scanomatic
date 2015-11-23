__author__ = 'martin'

from types import StringTypes
import numpy as np
from matplotlib import pyplot as plt

from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory


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