import glob
import numpy as np
import os
import re
import types

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scanomatic.generics.maths import mid50_mean
from scanomatic.io.image_data import ImageData
from scanomatic.io.image_loading import load_colony_images_for_animation
from scanomatic.io.logger import Logger
from scanomatic.io.movie_writer import MovieWriter
from scanomatic.io.paths import Paths
from scanomatic.io.pickler import unpickle_with_unpickler

# This import is used in 3D plotting just not explicitly stupid matplotlib

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
            (unpickle_with_unpickler(np.load, data) -
             mid50_mean(unpickle_with_unpickler(np.load, data)[unpickle_with_unpickler(np.load, bg)]))[
                unpickle_with_unpickler(np.load, blob)].sum()
            for data, blob, bg in zip(data_paths, blob_paths, background_paths)
        ])

    else:
        return np.array([
            unpickle_with_unpickler(np.load, data)[unpickle_with_unpickler(np.load, blob)].sum()
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


def animate_colony_growth(save_target, position, analysis_folder, fps=12, project_compilation=None, fig=None,
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
    data = unpickle_with_unpickler(np.load, files[0]).astype(np.float64)
    for i, ax in enumerate(fig.axes[:-1]):
        ims.append(ax.imshow(data, interpolation='nearest', vmin=0, vmax=(100 if i == 0 else 1)))

    @MovieWriter(save_target, "Colony detection animation", fps=fps, fig=fig)
    def _plotter():

        for idx, index in enumerate(image_indices):

            ims[0].set_data(unpickle_with_unpickler(np.load, files[idx]))
            base_name = files[idx][:-21]
            image_ax.set_title("Image (t={0:.1f}h)".format(
                image_indices[index] if interval is None else image_indices[index] * interval))

            for j, ending in enumerate(('.background.filter.npy', '.blob.filter.npy',
                                        '.blob.trash.current.npy', '.blob.trash.old.npy')):

                im_data = unpickle_with_unpickler(np.load, base_name + ending)
                if im_data.ndim == 2:
                    ims[j + 1].set_data(unpickle_with_unpickler(np.load, base_name + ending))

            set_axvspan_width(polygon, curve_times[idx])
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

    data = unpickle_with_unpickler(np.load, files[0])
    im = image_ax.imshow(data, interpolation='nearest', vmin=0, vmax=100)

    coords_x, coords_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    _, curve_times, polygon = plot_growth_curve(analysis_folder, position, curve_ax)
    curve_ax.set_ylabel("")
    curve_ax.set_xlabel("")

    _sqaure_ax(curve_ax)

    @MovieWriter(save_target, "Colony detection animation", fps=fps, fig=fig)
    def _plotter():

        ax3d = None

        for idx, index in enumerate(image_indices):

            im.set_data(unpickle_with_unpickler(np.load, files[idx]))

            # Added suffix length too
            base_name = files[idx][:-(10 + 11)]

            image_ax.set_title("Image (Time={0:.1f}h)".format(
                image_indices[index] if interval is None else image_indices[index] * interval))

            cells = unpickle_with_unpickler(np.load, base_name + ".image.cells.npy")
            if cells.ndim != 2:
                cells = np.zeros_like(coords_y)
            else:
                cells = np.round(cells * height_conversion, 1)

            if ax3d:
                fig.delaxes(ax3d)
            ax3d = fig.add_subplot(1, 3, 2, projection='3d')
            ax3d.view_init(elev=35., azim=(idx*rotation_speed) % 360)
            ax3d.set_axis_off()
            ax3d.plot_surface(coords_x, coords_y, cells, rstride=3, cstride=3, lw=.2, edgecolors="w")

            ax3d.set_xlim(xmin=0, xmax=coords_x.shape[0])
            ax3d.set_ylim(ymin=0, ymax=coords_x.shape[1])
            ax3d.set_zlim(zmin=0, zmax=15)

            _sqaure_ax(ax3d)

            set_axvspan_width(polygon, curve_times[idx])

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
            pos = tuple(np.random.randint(p_shape) for p_shape in data[0][plate].shape[:2])
            if ax_title is None:
                ax.set_title("Plate {0}, Pos {1}".format(plate, pos))

            ymin = None
            ymax = None
            for data_set in range(data_sets):
                curve_data = np.ma.masked_invalid(data[data_set][plate][pos])
                curves[data_set].set_ydata(curve_data)
                if ymin is None or curve_data.min() < ymin:
                    ymin = curve_data.min() * 0.9
                if ymax is None or curve_data.max() > ymax:
                    ymax = curve_data.max() * 1.2
            ax.set_ylim(ymin=ymin, ymax=ymax)
            elapsed += 1.0/fps
            yield

    return _plotter()
