from __future__ import absolute_import

from itertools import product
import pytest
from collections import namedtuple

from types import NoneType
import numpy as np
from numpy import ndarray

from scanomatic.image_analysis import grid_array as grid_array_module
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory

MockedGridCell = namedtuple('GridCell', ['xy1', 'xy2'])


def _get_grid_array_instance(im):
    image_identifier = [42, 1337]
    pinning = (8, 12)
    analysis_model = AnalysisModelFactory.create()
    analysis_model.output_directory = ""
    grid_array_instance = grid_array_module.GridArray(
        image_identifier, pinning, analysis_model)
    correction = (0, 0)
    grid_array_instance.detect_grid(im, grid_correction=correction)
    return grid_array_instance


@pytest.fixture(scope='session')
def grid_array(easy_plate):
    """Instantiate a GridArray object with a gridded image"""
    return _get_grid_array_instance(easy_plate)


@pytest.fixture(scope='session')
def bad_grid_array(hard_plate):
    return _get_grid_array_instance(hard_plate)


@pytest.mark.parametrize("grid_cell,expected_type", (
    (None, NoneType),
    (MockedGridCell(xy1=None, xy2=None), NoneType),
))
def test_get_im_slicei_no_im(grid_cell, expected_type):
    im = None
    im_slice = grid_array_module._get_image_slice(im, grid_cell)
    assert isinstance(im_slice, expected_type)


@pytest.mark.parametrize("grid_cell,expected_type,expected_shape", (
    (None, NoneType, None),
    (MockedGridCell(xy1=None, xy2=None), NoneType, None),
    (MockedGridCell(xy1=(1, 1), xy2=None), NoneType, None),
    (MockedGridCell(xy1=None, xy2=(1, 1)), NoneType, None),
    (MockedGridCell(xy1=(1, 1), xy2=(1,)), NoneType, None),
    (MockedGridCell(xy1=(1,), xy2=(1, 1)), NoneType, None),
    (MockedGridCell(xy1=(1, 1), xy2=(1, 1)), ndarray, (0, 0)),
    (MockedGridCell(xy1=(0, 1), xy2=(10, 20)), ndarray, (10, 19)),
    (
        MockedGridCell(xy1=np.array((0, 1)), xy2=np.array((10, 20))),
        ndarray,
        (10, 19)
    ),
))
def test_get_im_slice(easy_plate, grid_cell, expected_type, expected_shape):

    im_slice = grid_array_module._get_image_slice(easy_plate, grid_cell)
    assert isinstance(im_slice, expected_type)
    if expected_shape is not None:
        assert im_slice.shape == expected_shape


class TestGridDetection():

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

    def test_grid_cells_inherits_poly_coeffs(self):
        coeffs = [1, 1, 2, 3, 5]
        analysis_model = AnalysisModelFactory.create(
            cell_count_calibration=coeffs)

        grid_array_instance = grid_array_module.GridArray(
            [None, 0], (8, 12), analysis_model)
        grid_array_instance._init_grid_cells()

        assert all(
            a == b for a, b in zip(
                grid_array_instance[(0, 0)]._polynomial_coeffs,
                coeffs))

    def test_hard_plate_has_no_grid(self, bad_grid_array):
        """This tests a side effect of if gridding fails.

        Because the way the fixture works here there's no direct way.
        Basically, had a valid grid been detected the below
        field would have gotten a value.
        """
        assert bad_grid_array._grid_cell_size is None
