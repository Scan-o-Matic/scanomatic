import numpy as np
import pytest

from scanomatic.image_analysis import grid


class TestGetGridSpacings:

    def test_getting_spacing(self):

        x = np.array([1, 12, 10, 11])
        y = np.array([20, 80, 40, 42])

        spacings = grid.get_grid_spacings(x, y, 10, 20, leeway=0.01)
        assert spacings == (10.0, 20.0)

    @pytest.mark.parametrize('x,y,expect_x,expect_y,leeway,expected_outcome', [
        (
            np.array([1, 10]),
            np.array([0, 10]),
            10,
            10,
            0.01,
            (None, 10.0)
        ),
        (
            np.array([0, 10]),
            np.array([1, 10]),
            10,
            10,
            0.01,
            (10.0, None)
        ),
        (
            np.array([0, 10]),
            np.array([1, 10]),
            10,
            10,
            0.0,
            (None, None)
        ),
        (
            np.array([]),
            np.array([1, 10]),
            10,
            10,
            0.0,
            (None, None)
        ),
    ])
    def test_failing_spacings_returns_none(
            self, x, y, expect_x, expect_y, leeway, expected_outcome):

        spacings = grid.get_grid_spacings(
            x, y, expect_x, expect_y, leeway=leeway)
        assert spacings == expected_outcome


class TestGetGridParameters:

    def test_getting_grid_parameters(self):

        x = np.array([0, 11, 16, 21, 33, 34, 41, 51])
        y = np.array([1000, 1041, 1031, 1071, 1077, 1024, 1081, 1101])
        grid_shape = (4, 6)
        spacings = (10, 10)
        center, new_spacings = grid.get_grid_parameters(
            x, y, grid_shape, spacings=spacings)

        assert None not in center
        assert (x > center[0]).any()
        assert (x < center[0]).any()
        assert (y > center[1]).any()
        assert (y < center[1]).any()

        np.testing.assert_allclose(new_spacings, spacings, rtol=0.01)

    def test_failing_spacings_returns_nones(self):

        x = np.array([0, 25])
        y = np.array([1000, 1041, 1031, 1071, 1077, 1024, 1081, 1101])
        grid_shape = (4, 6)
        spacings = (10, 10)
        center, new_spacings = grid.get_grid_parameters(
            x, y, grid_shape, spacings=spacings)

        assert center is None
        assert new_spacings is None


class TestGetGrid:

    def test_getting_grid_easy_plate(self, easy_plate):
        """Expect grid and to be inside image.

        TODO: Expect proximity to curated positions
        """
        expected_spacings = (212, 212)
        expected_center = tuple([s / 2.0 for s in easy_plate.shape])
        validate_parameters = False
        grid_shape = (12, 8)
        grid_correction = None

        draft_grid, _, _, _, spacings, _ = grid.get_grid(
            easy_plate,
            expected_spacing=expected_spacings,
            expected_center=expected_center,
            validate_parameters=validate_parameters,
            grid_shape=grid_shape,
            grid_correction=grid_correction)

        assert draft_grid is not None
        assert draft_grid.shape == (2, ) + grid_shape

        assert spacings is not None
        for dim in range(2):
            d_spacing = spacings[dim]
            coord_components = draft_grid[dim]
            assert (coord_components > d_spacing / 2.0).all()
            assert (coord_components <
                    easy_plate.shape[dim] - d_spacing / 2.0).all()

    def test_getting_expecting_wrong_spacings_fails(self, easy_plate):

        expected_spacings = (137, 137)
        expected_center = tuple([s / 2.0 for s in easy_plate.shape])
        validate_parameters = False
        grid_shape = (12, 8)
        grid_correction = None

        draft_grid, _, _, _, spacings, _ = grid.get_grid(
            easy_plate,
            expected_spacing=expected_spacings,
            expected_center=expected_center,
            validate_parameters=validate_parameters,
            grid_shape=grid_shape,
            grid_correction=grid_correction)

        assert draft_grid is None

    def test_getting_grid_hard_plate(self, hard_plate):
        """Only expect to be a correctly shaped grid."""
        expected_spacings = (212, 212)
        expected_center = tuple([s / 2.0 for s in hard_plate.shape])
        validate_parameters = False
        grid_shape = (12, 8)
        grid_correction = None

        draft_grid, _, _, _, spacings, _ = grid.get_grid(
            hard_plate,
            expected_spacing=expected_spacings,
            expected_center=expected_center,
            validate_parameters=validate_parameters,
            grid_shape=grid_shape,
            grid_correction=grid_correction)

        assert draft_grid is not None
        assert draft_grid.shape == (2, ) + grid_shape

        assert spacings is not None
