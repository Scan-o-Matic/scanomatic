
import glob
import os
import re
import types
import numpy as np

from scanomatic.io.movie_writer import MovieWriter
from scanomatic.io.image_data import ImageData
from scanomatic.io.paths import Paths
from scanomatic.io.logger import Logger
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory
from scanomatic.generics.maths import mid50_mean

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

_pattern = re.compile(r".*_([0-9]+)_[0-9]+_[0-9]+_[0-9]+\..*")
_logger = Logger("Phenotype Results QC")
_marker_sequence = ['v', 'o', 's', '+', 'x', 'D', '*', '^']


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


def calculate_growth_curve(data_paths, blob_paths, background_paths=None):

    if background_paths is not None:
        return np.array([
            (np.load(data) - mid50_mean(np.load(data)[np.load(bg)]))[np.load(blob)].sum()
            for data, blob, bg in zip(data_paths, blob_paths, background_paths)
        ])

    else:
        return np.array([
            np.load(data)[np.load(blob)].sum()
            for data, blob in zip(data_paths, blob_paths)
        ])


def plot_growth_curve(growth_data, position, ax=None, save_target=None):

    if ax is None:
        ax = plt.figure().gca()

    times, data = ImageData.read_image_data_and_time(growth_data)

    ax.semilogy(times, data[position[0]][position[1], position[2]], "g-", basey=2)
    ax.set_xlim(xmin=0, xmax=times.max() + 1)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Population size [cells]")
    curve_times = ax.lines[0].get_data()[0]

    polygon = ax.axvspan(0, 0, color=(0, 1, 0, 0.5))
    polygon.xy = np.vstack((polygon.xy, polygon.xy[0].reshape(1, 2)))

    if save_target is not None:
        ax.figure.savefig(save_target)

    return ax, curve_times, polygon


def set_axvspan_width(polygon, width):

    polygon.xy[2:4, 0] = width


def load_colony_images_for_animation(analysis_directory, position, project_compilation=None, positioning="one-time"):
    """

    :param analysis_directory: path to analysis directory
    :type  analysis_directory: str
    :param position: list/tuple of colony to extract. (Plate, Row, Colum) Note that all positions should be
    enumerated from 1. because confusion!
    :type position: [int]
    :param project_compilation: Path to the associated compilation file, inferred if not submitted
    :type project_compilation: str
    :param positioning: Type of positioning to simulate. Default is "one-time", which uses the gridding image
    positioning all through.Use "detected" for the position actually detected.
    :type positioning: str
    :return: First array is a 1D time-vector, second array is a 3D image sequence vector where the last dimension
    is time, the third array is the 2D plate slice of the last image.
    :rtype : numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    # TODO: Verify that it is the correct image that is displayed

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

    assert positioning in ('one-time', 'detected'), "Unknown positioning argument"

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

    # grids number +1
    grid = np.load(os.path.join(analysis_directory, Paths().grid_pattern.format(position[0] + 1)))
    grid_size = np.load(os.path.join(analysis_directory, Paths().grid_size_pattern.format((position[0] + 1))))

    compilation_results = CompileImageAnalysisFactory.serializer.load(project_compilation)
    compilation_results = sorted(compilation_results, key=lambda e: e.image.index)

    times = np.array(tuple(entry.image.time_stamp for entry in compilation_results))
    images = np.zeros(tuple(grid_size) + times.shape, dtype=np.uint16)
    im = None
    ref_plate_model = compilation_results[-1].fixture.plates

    for i, entry in enumerate(compilation_results):
        try:
            im = plt.imread(entry.image.path)
        except IOError:
            im = plt.imread(os.path.join(experiment_directory, os.path.basename(entry.image.path)))

        if positioning == 'one-time':
            plate_model = ref_plate_model[position[0]]
        elif positioning == 'detected':
            plate_model = entry.fixture.plates[position[0]]
        else:
            raise ValueError("Positioning can't be '{0}'".format(positioning))

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
                          cmap=plt.cm.gray, colony_title=None, positioning="one-time"):

    _logger.info("Loading colony images")
    times, images, _ = load_colony_images_for_animation(analysis_folder, position,
                                                        project_compilation=project_compilation,
                                                        positioning=positioning)

    if fig is None:
        fig = plt.figure()

    im_ax = fig.add_subplot(1, 2, 1)
    curve_ax = fig.add_subplot(1, 2, 2)
    im = im_ax.imshow(images[..., 0], interpolation="nearest", vmin=images.min(), vmax=images.max(), cmap=cmap)

    _, curve_times, polygon = plot_growth_curve(analysis_folder, position, curve_ax)

    fig.tight_layout(h_pad=0.5)

    @MovieWriter(save_target, "Colony growth animation", fps=fps, fig=fig)
    def _plotter():

        for i, time in enumerate(times):

            im_ax.set_title(
                colony_title if colony_title is not None else
                "Pos {0}, (t={1:.1f}h)".format(position, time / 3600.))
            im.set_data(images[..., i])
            set_axvspan_width(polygon, curve_times[i])
            _sqaure_ax(curve_ax)
            yield

    return _plotter()


def detection_files(data_pos, source_location=None, suffix=".calibrated.image.npy"):

    if source_location is None:
        source_location = Paths().log

    pattern = os.path.join(source_location, "grid_cell_*_{0}_{1}_{2}".format(*data_pos) + suffix)
    files = np.array(glob.glob(pattern))
    image_indices = [int(_pattern.match(f).groups()[0]) for f in files]
    index_order = np.argsort(image_indices)
    return files[index_order], np.array(image_indices)[index_order]


def animate_blob_detection(save_target, position, analysis_folder,
                           fig=None, fps=12, interval=None):

    if fig is None:
        fig = plt.figure()

    files, image_indices = detection_files(position, analysis_folder)

    titles = ["Image", "Background", "Blob", "Trash (Now)", "Trash (Previous)", "Growth Data"]
    axes = len(titles)
    if len(fig.axes) != axes:
        fig.clf()
        for i in range(axes):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.set_title(titles[i])

    curve_ax, curve_times, polygon = plot_growth_curve(analysis_folder, position, fig.axes[-1])

    image_ax = fig.axes[0]
    ims = []
    data = np.load(files[0]).astype(np.float64)
    for i, ax in enumerate(fig.axes[:-1]):
        ims.append(ax.imshow(data, interpolation='nearest', vmin=0, vmax=(100 if i == 0 else 1)))

    @MovieWriter(save_target, "Colony detection animation", fps=fps, fig=fig)
    def _plotter():

        for i, index in enumerate(image_indices):

            ims[0].set_data(np.load(files[i]))
            base_name = files[i][:-21]
            image_ax.set_title("Image (t={0:.1f}h)".format(
                image_indices[index] if interval is None else image_indices[index] * interval))

            for j, ending in enumerate(('.background.filter.npy', '.blob.filter.npy',
                                        '.blob.trash.current.npy', '.blob.trash.old.npy')):

                im_data = np.load(base_name + ending)
                if im_data.ndim == 2:
                    ims[j + 1].set_data(np.load(base_name + ending))

            set_axvspan_width(polygon, curve_times[i])
            _sqaure_ax(curve_ax)

            yield

    return _plotter()


def animate_3d_colony(save_target, position, analysis_folder,
                      fig=None, fps=12, interval=None, height_conversion=.001, rotation_speed=5.):

    if fig is None:
        fig = plt.figure(figsize=(10, 3))

    files, image_indices = detection_files(position, analysis_folder)

    titles = ["Image", "3D", "Population Size [cells]"]
    axes = len(titles)
    if len(fig.axes) != axes:
        fig.clf()
        for i in range(axes):
            if i != 1:
                ax = fig.add_subplot(1, 3, i + 1)
                ax.set_title(titles[i])

    image_ax, curve_ax = fig.axes

    data = np.load(files[0])
    im = image_ax.imshow(data, interpolation='nearest', vmin=0, vmax=100)

    coords_x, coords_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    _, curve_times, polygon = plot_growth_curve(analysis_folder, position, curve_ax)
    curve_ax.set_ylabel("")
    curve_ax.set_xlabel("")

    _sqaure_ax(curve_ax)

    @MovieWriter(save_target, "Colony detection animation", fps=fps, fig=fig)
    def _plotter():

        ax3d = None

        for i, index in enumerate(image_indices):

            im.set_data(np.load(files[i]))

            # Added suffix length too
            base_name = files[i][:-(10 + 11)]

            image_ax.set_title("Image (Time={0:.1f}h)".format(
                image_indices[index] if interval is None else image_indices[index] * interval))

            cells = np.load(base_name + ".image.cells.npy")
            if cells.ndim != 2:
                cells = np.zeros_like(coords_y)
            else:
                cells = np.round(cells * height_conversion, 1)

            if ax3d:
                fig.delaxes(ax3d)
            ax3d = fig.add_subplot(1, 3, 2, projection='3d')
            ax3d.view_init(elev=35., azim=(i*rotation_speed) % 360)
            ax3d.set_axis_off()
            ax3d.plot_surface(coords_x, coords_y, cells, rstride=3, cstride=3, lw=.2, edgecolors="w")

            ax3d.set_xlim(xmin=0, xmax=coords_x.shape[0])
            ax3d.set_ylim(ymin=0, ymax=coords_x.shape[1])
            ax3d.set_zlim(zmin=0, zmax=15)

            _sqaure_ax(ax3d)

            set_axvspan_width(polygon, curve_times[i])

            yield

    return _plotter()


def animate_example_curves(save_target, growth_data=None, fig=None, fps=2, ax_title=None, duration=4, legend=None,
                           cmap=plt.cm.terrain, **kwargs):

    if isinstance(growth_data, types.StringType):
        growth_data = [growth_data]

    if fig is None:
        fig = plt.figure()

    times = []
    data = []

    for path in growth_data:
        t, d = ImageData.read_image_data_and_time(path)
        times.append(t)
        data.append(d)

    ax = plt.gca()
    ax.set_title(ax_title)

    curves = []
    data_sets = len(data)
    for i in range(data_sets):
        curve = ax.semilogy(times[i], data[i][0][0, 0], marker=_marker_sequence[i], basey=2,
                            color=cmap(i/float(data_sets)), mec=cmap(i/float(data_sets)), **kwargs)[0]
        curves.append(curve)

    ax.set_xlim(xmin=0, xmax=max(np.array(t).max() for t in times) + 0.5)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Population Size [cells]")
    if ax_title is not None:
        ax.set_title(ax_title)
    if legend:
        ax.legend(legend, loc='lower right')

    @MovieWriter(save_target, "Example Curves", fps=fps, fig=fig)
    def _plotter():

        elapsed = 0

        while elapsed < duration:
            plate = np.random.randint(data[0].shape[0])
            pos = tuple(np.random.randint(d) for d in data[0][plate].shape[:2])
            if ax_title is None:
                ax.set_title("Plate {0}, Pos {1}".format(plate, pos))

            ymin = None
            ymax = None
            for i in range(data_sets):
                d = np.ma.masked_invalid(data[i][plate][pos])
                curves[i].set_ydata(d)
                if ymin is None or d.min() < ymin:
                    ymin = d.min() * 0.9
                if ymax is None or d.max() > ymax:
                    ymax = d.max() * 1.2
            ax.set_ylim(ymin=ymin, ymax=ymax)
            elapsed += 1.0/fps
            yield

    return _plotter()
