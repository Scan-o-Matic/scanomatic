from itertools import product
import pytest

from scipy import ndimage

from scanomatic.image_analysis import grid_array as grid_array_module
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.data_processing.calibration import (
    get_polynomial_coefficients_from_ccc,
    __CCC as CCC,
    get_empty_ccc,
    CellCountCalibration
)
from scanomatic.models.analysis_model import get_original_calibration

@pytest.fixture(scope="class")
def grid_array():
    """Instantiate a GridArray object with a gridded image"""
    image_identifier = [42, 1337]
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


@pytest.fixture(scope="class")
def grid_array_using_ccc():
    image_identifier = [42, 1337]
    pinning = (8, 12)
    analysis_model = AnalysisModelFactory.create(
        cell_count_calibration="TEST")
    analysis_model.output_directory = ""
    image = ndimage.io.imread(
        './scanomatic/image_analysis/test/testdata/test_fixture_easy.tiff')
    grid_array_instance = grid_array_module.GridArray(
        image_identifier, pinning, analysis_model)
    correction = (0, 0)
    grid_array_instance.detect_grid(image, grid_correction=correction)
    return grid_array_instance


class TestGridDetection():

    def setup_method(self):
        global CCC
        CCC.clear()
        ccc = get_empty_ccc('Test', 'CCC')
        self._ccc_id =  ccc[CellCountCalibration.identifier]
        ccc[CellCountCalibration.polynomial] = {
            'test': {'power': 5, 'coefficients': [10, 0, 0, 0, 150, 0]}
        }

        ccc[CellCountCalibration.deployed_polynomial] = 'test'
        CCC[ccc[CellCountCalibration.identifier]] = ccc

    def teardown_method(self):
        global CCC
        del CCC[self._ccc_id]

    def test_grid_shape_is_correct(self, grid_array):
        pinning = (8, 12)
        assert grid_array.grid_shape == pinning[:: -1]

    def test_grid_centres_inside_xy1_xy2(self, grid_array):

        def fail_text():
            return "Centre out of bounds for cell ({}, {})".format(row, col)

        rows, cols = grid_array.grid_shape
        for row, col in product(range(rows), range(cols)):
            grid_cell = grid_array[(row, col)]
            assert grid_cell.xy1[0] < grid_array.grid[0][row][col], fail_text()
            assert grid_cell.xy1[1] < grid_array.grid[1][row][col], fail_text()
            assert grid_cell.xy2[0] > grid_array.grid[0][row][col], fail_text()
            assert grid_cell.xy2[1] > grid_array.grid[1][row][col], fail_text()

    def test_gridarray_and_gridcell_positions_same(self, grid_array):

        def fail_text():
            return "Positions not same for cell ({}, {})".format(row, col)

        rows, cols = grid_array.grid_shape
        for row, col in product(range(rows), range(cols)):
            grid_cell = grid_array[(row, col)]
            assert grid_cell.position == (row, col), fail_text()

    def test_gridcells_inherit_gridarray_identifier(self, grid_array):

        def fail_text():
            return "Identifier not same for cell ({}, {})".format(row, col)

        image_identifier = [grid_array.image_index, grid_array.index]
        assert image_identifier == [42, 1337]

        rows, cols = grid_array.grid_shape
        for row, col in product(range(rows), range(cols)):
            grid_cell = grid_array[(row, col)]
            assert grid_cell._identifier[0] == image_identifier, fail_text()

    def test_grid_cells_uses_default_polynomial(self, grid_array):

        assert (
            grid_array[(0, 0)]._polynomial_coeffs ==
            get_original_calibration())


    def test_grid_cells_uses_specified_ccc(self, grid_array_using_ccc):

        assert all(
            a == b for a, b in zip(
                grid_array_using_ccc[(0, 0)]._polynomial_coeffs,
                get_polynomial_coefficients_from_ccc(self._ccc_id)))
