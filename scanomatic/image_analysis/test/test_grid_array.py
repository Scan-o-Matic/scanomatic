from itertools import product
import pytest

from scipy import ndimage

from scanomatic.image_analysis import grid_array as grid_array_module
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory


@pytest.fixture(scope="class")
def grid_array():
    """Instantiate a GridArray object with a gridded image"""
    image_identifier = (0, 1)
    pinning = (8, 12)
    analysis_model = AnalysisModelFactory.create()
    analysis_model.output_directory = ""
    image = ndimage.io.imread(
        './scanomatic/image_analysis/test/testdata/test_fixture_easy.tiff')
    grid_array_instance = grid_array_module.GridArray(
        image_identifier, pinning, analysis_model)
    correction = (0, 0)
    grid_array_instance.detect_grid(image, grid_correction=correction)
    return grid_array_instance


class TestGridDetection():

    def test_grid_centres_inside_xy1_xy2(self, grid_array):
        rows, cols = grid_array.grid_shape
        for row, col in product(range(rows), range(cols)):
            grid_cell = grid_array[(row, col)]
            try:
                assert grid_cell.xy1[0] < grid_array.grid[0][row][col]
                assert grid_cell.xy1[1] < grid_array.grid[1][row][col]
                assert grid_cell.xy2[0] > grid_array.grid[0][row][col]
                assert grid_cell.xy2[1] > grid_array.grid[1][row][col]
            except Exception as err:
                raise Exception("{} for cell ({}, {})".format(err, row, col))


    def test_grid_shape_is_correct(self, grid_array):
        pinning = (8, 12)
        assert grid_array.grid_shape == pinning[:: -1]
