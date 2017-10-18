import glob
import os
import numpy as np
from scanomatic.io.paths import Paths
from scanomatic.image_analysis.image_basics import load_image_to_numpy
from scanomatic.io.logger import Logger
from scanomatic.io.pickler import unpickle_with_unpickler
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory
from scanomatic.generics.purge_importing import ExpiringModule

_logger = Logger("Analysis Utils")


def produce_grid_images(path=".", plates=None, image=None, mark_position=None, compilation=None):

    project_path = os.path.join(os.path.dirname(os.path.abspath(path)))

    if plates:
        plates = [plate_index + 1 for plate_index in plates]

    if compilation:
        if not os.path.isfile(compilation):
            raise ValueError("There's no compilation at {0}".format(compilation))
    else:
        for compilation_pattern in (Paths().project_compilation_pattern,
                                    Paths().project_compilation_from_scanning_pattern,
                                    Paths().project_compilation_from_scanning_pattern_old):

            compilations = glob.glob(
                os.path.join(os.path.dirname(os.path.abspath(path)), compilation_pattern.format("*")))

            if compilations:
                break

        if not compilations:
            raise ValueError("There are no compilations in the parent directory")

        compilation = compilations[0]

    _logger.info("Using '{0}' to produce grid images".format(os.path.basename(compilation)))

    compilation = CompileImageAnalysisFactory.serializer.load(compilation)

    image_path = compilation[-1].image.path
    all_plates = compilation[-1].fixture.plates
    if image is not None:
        for c in compilation:
            if os.path.basename(c.image.path) == os.path.basename(image):
                image_path = c.image.path
                all_plates = c.fixture.plates
                break

    try:
        image = load_image_to_numpy(image_path, dtype=np.uint8)
    except IOError:

        try:
            image = load_image_to_numpy(os.path.join(project_path, os.path.basename(image_path)), dtype=np.uint8)
        except IOError:
            raise ValueError("Image doesn't exist, can't show gridding")

    _logger.info("Producing grid images for {0}".format(plates))

    for plate in all_plates:

        if plate is not None and plate.index not in plates:
            continue

        plate_image = image[plate.y1: plate.y2, plate.x1: plate.x2]

        grid_path = os.path.join(
            path, Paths().grid_pattern.format(plate.index))
        try:
            grid = unpickle_with_unpickler(np.load, grid_path)
        except IOError:
            _logger.warning("Could not find any grid: " + grid_path)
            grid = None

        image_path = Paths().experiment_grid_image_pattern.format(plate.index)
        make_grid_im(plate_image, grid, os.path.join(path, image_path),
                     marked_position=mark_position)


def make_grid(im, grid_plot, grid, marked_position):

        x = 0
        y = 1
        for row in range(grid.shape[1]):

            grid_plot.plot(
                grid[x, row, :], -grid[y, row, :] + im.shape[y], 'r-')

        for col in range(grid.shape[2]):

            grid_plot.plot(
                grid[x, :, col], -grid[y, :, col] + im.shape[y], 'r-')

        if marked_position is None:
            marked_position = (-1, 0)

        grid_plot.plot(
            grid[x, marked_position[0], marked_position[1]],
            grid[y, marked_position[0], marked_position[1]] +
            im.shape[y],
            'o', alpha=0.75, ms=10, mfc='none', mec='blue', mew=1)


def make_grid_im(im, grid, save_grid_name, marked_position=None):

    with ExpiringModule("matplotlib", run_code="mod.use('Svg')"):
        with ExpiringModule("matplotlib.pyplot") as plt:

            grid_image = plt.figure()
            grid_plot = grid_image.add_subplot(111)
            grid_plot.imshow(im.T, cmap=plt.cm.gray)

            if grid is not None:
                make_grid(im, grid_plot, grid, marked_position)

            ax = grid_image.gca()
            ax.set_xlim(0, im.shape[0])
            ax.set_ylim(0, im.shape[1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            grid_image.savefig(
                save_grid_name,
                pad_inches=0.01,
                format='svg',
                bbox_inches='tight')
