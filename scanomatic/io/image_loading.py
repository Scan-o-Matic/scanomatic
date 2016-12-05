import glob
import numpy as np
import os

from scanomatic.io.paths import Paths
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory
from scanomatic.io import logger
from scanomatic.io.pickler import unpickle_with_unpickler
from scanomatic.image_analysis.image_basics import load_image_to_numpy

_logger = logger.Logger("Image loader")


def _get_project_compilation(analysis_directory, file_name=None):

    experiment_directory = os.sep.join(analysis_directory.split(os.sep)[:-1])
    if file_name:
        project_compilation = os.path.join(experiment_directory, file_name)
    else:
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

    return project_compilation


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


def slice_im(plate_im, colony_position, colony_size):

    lbound = colony_position - np.floor(colony_size / 2)
    ubound = colony_position + np.ceil(colony_size / 2)
    if (ubound - lbound != colony_size).any():
        ubound += colony_size - (ubound - lbound)

    return plate_im[lbound[0]: ubound[0], lbound[1]: ubound[1]]


def _load_grid_info(analysis_directory, plate):
    # grids number +1
    grid = unpickle_with_unpickler(
        np.load, os.path.join(analysis_directory, Paths().grid_pattern.format(plate + 1)))
    grid_size = unpickle_with_unpickler(
        np.load, os.path.join(analysis_directory, Paths().grid_size_pattern.format((plate + 1))))
    return grid, grid_size


def load_colony_image(position, compilation_result=None, analysis_directory=None, time_index=None,
                      compilation_file_name=None, experiment_directory=None, grid=None, grid_size=None):

    if not compilation_result:
        compilation_file = _get_project_compilation(analysis_directory, file_name=compilation_file_name)
        compilation_result = CompileImageAnalysisFactory.serializer.load(compilation_file)[time_index]
        if not experiment_directory:
            experiment_directory = os.path.dirname(compilation_file)

    try:
        im = load_image_to_numpy(compilation_result.image.path, dtype=np.uint8)
    except IOError:
        im = load_image_to_numpy(os.path.join(experiment_directory, os.path.basename(compilation_result.image.path)),
                                 dtype=np.uint8)

    if grid is None or grid_size is None:
        grid, grid_size = _load_grid_info(analysis_directory, position[0])

    plate_model = compilation_result.fixture.plates[position[0]]
    x = sorted((plate_model.x1, plate_model.x2))
    y = sorted((plate_model.y1, plate_model.y2))

    y, x = _bound(im.shape, y, x)

    # As gridding is done on plates as seen in the scanner while plate positioning is done on plates
    # as seen by the scanner the inverse direction of the short dimension is needed and needed after
    # slicing out the plate
    im = im[y[0]: y[1], x[0]: x[1]][:, ::-1]

    return slice_im(im, grid[:, grid.shape[1] - position[2] - 1, position[1]], grid_size)


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

    assert positioning in ('one-time', 'detected'), "Unknown positioning argument"

    analysis_directory = os.path.abspath(analysis_directory)

    if not project_compilation:
        project_compilation = _get_project_compilation(analysis_directory)

    experiment_directory = os.path.dirname(project_compilation)

    grid, grid_size = _load_grid_info(analysis_directory, position[0])

    compilation_results = CompileImageAnalysisFactory.serializer.load(project_compilation)
    compilation_results = sorted(compilation_results, key=lambda e: e.image.index)

    times = np.array(tuple(entry.image.time_stamp for entry in compilation_results))
    images = np.zeros(tuple(grid_size) + times.shape, dtype=np.uint16)
    im = None
    ref_plate_model = compilation_results[-1].fixture.plates

    for i, entry in enumerate(compilation_results):

        if positioning == 'one-time':
            entry.fixture.plates = tuple(
                ref_plate_model[position[0]] if j == position[0] else p for j, p in enumerate(entry.fixture.plates))
        elif positioning == 'detected':
            pass
        else:
            raise ValueError("Positioning can't be '{0}'".format(positioning))

        images[..., i] = load_colony_image(position, grid=grid, grid_size=grid_size, compilation_result=entry,
                                           experiment_directory=experiment_directory)

    return times, images, im
