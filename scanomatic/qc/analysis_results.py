
import glob
import os
import re

from scanomatic.io.movie_writer import *
from scanomatic.io.image_data import ImageData
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory

_pattern = re.compile(r".*_([0-9]+)_[0-9]+_[0-9]+_[0-9]+\.image.npy")
_logger = Logger("Phenotype Results QC")


def _sqaure_ax(ax):
    fig = ax.figure

    fig_width = fig.get_figwidth()
    fig_height = fig.get_figheight()

    extents = ax.get_position()
    ax_width = fig_width * extents.width
    ax_height = fig_height * extents.height
    if ax_width > ax_height:
        delta = (ax_width - ax_height) / (fig_width * 2.0)
        extents.x0 += delta
        extents.x1 -= delta
    elif ax_height > ax_width:
        delta = (ax_height - ax_width) / (fig_height * 2.0)
        extents.y0 += delta
        extents.y1 -= delta

    ax.set_position(extents)


def plot_growth_curve(growth_data, position, ax=None):

    if ax is None:
        ax = plt.figure().gca()

    times, data = ImageData.read_image_data_and_time(growth_data)

    ax.semilogy(times, data[position[0] - 1][position[1:]], "g-", basey=2)
    ax.set_xlim(xmin=0, xmax=times.max() + 1)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Population size [cells]")
    curve_times = ax.lines[0].get_data()[0]

    polygon = ax.axvspan(0, 0, color=(0, 1, 0, 0.5))
    polygon.xy = np.vstack((polygon.xy, polygon.xy[0].reshape(1,2)))

    return ax, curve_times, polygon


def set_axvspan_width(polygon, width):

    polygon.xy[2:4, 0] = width


def load_colony_images_for_animation(analysis_directory, position, project_compilation=None):
    """

    :param analysis_directory: path to analysis directory
    :type  analysis_directory: str
    :param position: list/tuple of colony to extract. (Plate, Row, Colum) Note that all positions should be
    enumerated from 1. because confusion!
    :type position: [int]
    :param project_compilation: Path to the associated compilation file, inferred if not submitted
    :type project_compilation: str
    :return: First array is a 1D time-vector, second array is a 3D image sequence vector where the last dimension
    is time, the third array is the 2D plate slice of the last image.
    :rtype : numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    def _bound(bounds, a, b):

        def bounds_check(bound, val):

            if 0 <= val < bound:
                return val
            elif val < 0:
                return 0
            else:
                return bound - 1

        return ((bounds_check(bounds[0], a[0]),
                 bounds_check(bounds[0], a[1])),
                (bounds_check(bounds[1], b[0]),
                 bounds_check(bounds[1], b[1])))

    def slice_im(plate, colony_position, colony_size):

        lbound = colony_position - np.floor(colony_size / 2)
        ubound = colony_position + np.ceil(colony_size / 2)
        if (ubound - lbound != colony_size).any():
            ubound += colony_size - (ubound - lbound)

        return plate[lbound[0]: ubound[0], lbound[1]:ubound[1]]

    plate_as_index = position[0] - 1
    analysis_directory = os.path.abspath(analysis_directory)

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
                raise ValueError()
            elif len(candidates) != 1:
                _logger.error(
                    "Found several project.compilation files in '{0}', unsure which to use.".format(
                            experiment_directory) +
                    "Either remove one of {0} or specify compilation-file in function call".format(candidates))
                raise ValueError()

            project_compilation = candidates[0]
    else:
        experiment_directory = os.path.dirname(project_compilation)

    grid = np.load(os.path.join(analysis_directory, Paths().grid_pattern.format(position[0])))
    grid_size = np.load(os.path.join(analysis_directory, Paths().grid_size_pattern.format((position[0]))))

    compilation_results = tuple(CompileImageAnalysisFactory.serializer.load(project_compilation))
    compilation_results = sorted(compilation_results, key=lambda entry: entry.image.index)

    times = np.array(tuple(entry.image.time_stamp for entry in compilation_results))
    images = np.zeros(tuple(grid_size) + times.shape, dtype=np.uint16)
    im = None

    for i, entry in enumerate(compilation_results):
        try:
            im = plt.imread(entry.image.path)
        except IOError:
            im = plt.imread(os.path.join(experiment_directory, os.path.basename(entry.image.path)))

        plate_model = entry.fixture.plates[plate_as_index]

        x = sorted((plate_model.x1, plate_model.x2))
        y = sorted((plate_model.y1, plate_model.y2))

        y, x = _bound(im.shape, y, x)

        # As gridding is done on plates as seen in the scanner while plate positioning is done on plates
        # as seen by the scanner the inverse direction of the short dimension is needed and needed after
        # slicing out the plate
        im = im[y[0]: y[1], x[0]: x[1]][:, ::-1]

        images[..., i] = slice_im(im, grid[:, position[2], position[1]], grid_size)

    return times, images, im


def animate_colony_growth(save_target, analysis_folder, position=(0, 0, 0), fps=12, project_compilation=None, fig=None,
                          cmap=plt.cm.gray):

    _logger.info("Loading colony images")
    times, images, _ = load_colony_images_for_animation(analysis_folder, position,
                                                        project_compilation=project_compilation)

    if fig is None:
        fig = plt.figure()

    im_ax = fig.add_subplot(1, 2, 1)
    curve_ax = fig.add_subplot(1, 2, 2, aspect=1.0)
    im = im_ax.imshow(images[..., 0], interpolation="nearest", vmin=images.min(), vmax=images.max(), cmap=cmap)

    _, curve_times, polygon = plot_growth_curve(analysis_folder, position, curve_ax)

    fig.tight_layout(h_pad=0.5)

    @Write_Movie(save_target, "Colony growth animation", fps=fps, fig=fig)
    def _plotter():

        for i, time in enumerate(times):

            im_ax.set_title("Pos {0}, (t={1:.1f}h)".format(position, time / 3600.))
            im.set_data(images[..., i])
            set_axvspan_width(polygon, curve_times[i])
            _sqaure_ax(curve_ax)
            yield


def animate_blob_detection(save_target, position=(0, 0, 0), source_location=None, growth_data=None,
                           fig=None, fps=3, interval=None):

    if source_location is None:
        source_location = Paths().log

    if fig is None:
        fig = plt.figure()

    pattern = os.path.join(source_location, "grid_cell_*_{0}_{1}_{2}.image.npy".format(*position))
    files = np.array(glob.glob(pattern))
    image_indices = [int(_pattern.match(f).groups()[0]) for f in files]
    index_order = np.argsort(image_indices)

    titles = ["Image", "Background", "Blob", "Trash (Now)", "Trash (Previous)", "Growth Data"]
    axes = len(titles)
    if len(fig.axes) != axes:
        fig.clf()
        for i in range(axes):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.set_title(titles[i])

    _, curve_times, polygon = plot_growth_curve(growth_data, position, fig.axes[-1])

    image_ax = fig.axes[0]
    ims = []
    data = np.load(files[0])
    for i, ax in enumerate(fig.axes):
        ims.append(ax.imshow(data, interpolation='nearest', vmin=0, vmax=(3000 if i == 0 else 1)))

    @Write_Movie(save_target, "Colony detection animation", fps=fps, fig=fig)
    def _plotter():

        for i in index_order:

            ims[0].set_data(np.load(files[i]))
            base_name = files[i][:-10]
            image_ax.set_title("Image (t={0:.1f})".format(
                image_indices[i] if interval is None else image_indices[i] * interval))

            for i, ending in enumerate(('.background.filter.npy', '.blob.filter.npy',
                              '.blob.trash.current.npy', '.blob.trash.old.npy')):

                ims[i + 1].set_data(np.load(base_name + ending))
                set_axvspan_width(polygon, curve_times[i])

            yield