from __future__ import absolute_import

from collections import namedtuple
import json
import os
import tempfile

from mock import MagicMock
import numpy as np
import pytest

from scanomatic.data.calibrationstore import CalibrationStore
from scanomatic.data_processing import calibration
from scanomatic.io import ccc_data
from scanomatic.io.ccc_data import (
    CalibrationEntryStatus, CCCImage, CCCMeasurement, CCCPlate, CCCPolynomial,
)
from scanomatic.io.paths import Paths
from tests.factories import make_calibration

# The curve fitting is done with coefficients as e^x, some tests
# want to have that expression to be zero while still real.
# Then this constant is used.
EVAL_AS_ZERO = -99

# Needs adding for all ccc_* paths if tests extended
# Don't worry Paths is a singleton
Paths().ccc_folder = os.path.join(tempfile.gettempdir(), 'tempCCC')
Paths().ccc_file_pattern = os.path.join(
    Paths().ccc_folder, '{0}.ccc')


@pytest.fixture
def store():
    return MagicMock(CalibrationStore)


def test_get_im_slice():
    """Test that _get_im_slice handles floats"""
    image = np.arange(0, 42).reshape((6, 7))
    model_tuple = namedtuple("Model", ['x1', 'x2', 'y1', 'y2'])
    model = model_tuple(1.5, 3.5, 2.5, 4.5)
    assert calibration._get_im_slice(image, model).sum() == 207


def test_poly_as_text():
    assert (
        calibration.poly_as_text([2, 0, -1, 0]) ==
        "y = -1.00E+00 x^1 + 2.00E+00 x^3"
    )


def test_get_calibration_polynomial_residuals():
    colony_summer = calibration.get_calibration_optimization_function(2)
    data = calibration.CalibrationData(
        source_value_counts=[[10, 2], [3]],
        source_values=[[1, 2], [3]],
        target_value=np.array([100, 126])
    )
    c1 = 0  # e^x = 1
    c2 = EVAL_AS_ZERO  # e^x = 0
    residuals = calibration.get_calibration_polynomial_residuals(
        [c2, c1],
        colony_summer,
        data,
    )
    assert (residuals == (100.0 - 14.0, 126.0 - 9.0)).all()


class TestGetCalibrationOptimizationFunction:

    def test_returns_expected_values(self):
        colony_summer = calibration.get_calibration_optimization_function(2)
        data = calibration.CalibrationData(
            source_value_counts=[[10, 2], [3]],
            source_values=[[1, 2], [3]],
            target_value=[100, 126]
        )
        c1 = np.log(2)  # e^x = 2
        c2 = np.log(4)  # e^x = 4
        sums = colony_summer(data, c2, c1)
        assert all(
            calc == target for calc, target in zip(sums, data.target_value)
        )

    @pytest.fixture(scope='session')
    def poly2(self):
        return calibration.get_calibration_optimization_function(2)

    @pytest.fixture(scope='session')
    def poly4(self):
        return calibration.get_calibration_optimization_function(4)

    @pytest.mark.parametrize("calibration_data,coeffs,expected", (
        (
            calibration.CalibrationData([[2], [3]], [[1], [1]], []),
            (0, 0),
            (6, 12),
        ),
        (
            calibration.CalibrationData([[2], [2]], [[1], [2]], []),
            (np.log(2), EVAL_AS_ZERO),
            (8, 16),
        ),
        (
            calibration.CalibrationData([[2], [1]], [[1], [1]], []),
            (EVAL_AS_ZERO, np.log(2)),
            (4, 2),
        ),
    ))
    def test_calibration_curve_fit_polynomial_function_many_colonies(
        self, calibration_data, coeffs, expected, poly2
    ):
        assert poly2(calibration_data, *coeffs) == expected

    @pytest.mark.parametrize("calibration_data,coeffs,expected", (
        (
            calibration.CalibrationData([[1]], [[1]], []),
            (np.log(2), EVAL_AS_ZERO, EVAL_AS_ZERO, EVAL_AS_ZERO),
            (2,),
        ),
        (
            calibration.CalibrationData([[1]], [[1]], []),
            (EVAL_AS_ZERO, np.log(2), EVAL_AS_ZERO, EVAL_AS_ZERO),
            (2,),
        ),
        (
            calibration.CalibrationData([[1]], [[1]], []),
            (EVAL_AS_ZERO, EVAL_AS_ZERO, np.log(2), EVAL_AS_ZERO),
            (2,),
        ),
        (
            calibration.CalibrationData([[1]], [[1]], []),
            (EVAL_AS_ZERO, EVAL_AS_ZERO, EVAL_AS_ZERO, np.log(2)),
            (2,),
        ),
        (
            calibration.CalibrationData([[2]], [[1]], []),
            (0, EVAL_AS_ZERO, EVAL_AS_ZERO, EVAL_AS_ZERO),
            (16,),
        ),
        (
            calibration.CalibrationData([[2]], [[1]], []),
            (EVAL_AS_ZERO, 0, EVAL_AS_ZERO, EVAL_AS_ZERO),
            (8,),
        ),
        (
            calibration.CalibrationData([[2]], [[1]], []),
            (EVAL_AS_ZERO, EVAL_AS_ZERO, 0, EVAL_AS_ZERO),
            (4,),
        ),
        (
            calibration.CalibrationData([[2]], [[1]], []),
            (EVAL_AS_ZERO, EVAL_AS_ZERO, EVAL_AS_ZERO, 0),
            (2,),
        ),
    ))
    def test_calibration_curve_fit_polynomial_function_each_coeff(
        self, calibration_data, coeffs, expected, poly4
    ):
        assert poly4(calibration_data, *coeffs) == pytest.approx(expected)

    @pytest.mark.parametrize('x, coeffs, test_coeffs', (
        (5, [1, 1, 0], [0, 0]),
        (
            6,
            [42, 0, 0, 7, 0],
            [np.log(42), EVAL_AS_ZERO, EVAL_AS_ZERO, np.log(7)],
        ),
    ))
    def test_calibration_functions_give_equal_results(
        self, x, coeffs, test_coeffs
    ):

        poly_fitter = calibration.get_calibration_optimization_function(
            len(test_coeffs))
        poly = calibration.get_calibration_polynomial(coeffs)

        assert poly(x) == pytest.approx(
            poly_fitter(
                calibration.CalibrationData([[x]], [[1]], [0]),
                *test_coeffs)[0])


class TestAccessToken:
    IDENTIFIER = 'foobar'
    ACCESS_TOKEN = 'password'

    @pytest.fixture
    def store(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            identifier=self.IDENTIFIER,
            active=False,
            access_token=self.ACCESS_TOKEN,
        )
        return store

    def test_missing_token(self, store):
        assert not calibration.is_valid_edit_request(
            store, self.IDENTIFIER
        ), "Edit request worked, despite missing token"

    def test_bad_token(self, store):
        assert not calibration.is_valid_edit_request(
            store, self.IDENTIFIER, access_token='bad'
        ), "Edit request worked, despite bad token"

    def test_valid_token(self, store):
        assert calibration.is_valid_edit_request(
            store, self.IDENTIFIER, access_token=self.ACCESS_TOKEN,
        ) is True, "Edit request failed, despite valid token."


class TestValidatePolynomial:

    @pytest.mark.parametrize('slope,p_value,stderr,expected', (
        (0.99, 0.01, 0.001, calibration.CalibrationValidation.OK),
        (0.89, 0.01, 0.001, calibration.CalibrationValidation.BadSlope),
        (0.99, 0.5, 0.001, calibration.CalibrationValidation.BadStatistics),
        (0.99, 0.01, 0.06, calibration.CalibrationValidation.BadStatistics),
    ))
    def test_validate_bad_correlation(self, slope, p_value, stderr, expected):

        assert calibration.validate_polynomial(
            slope, p_value, stderr) == expected


class TestGettingActiveCCCs:

    def test_unknown_ccc_raises_exception_for_poly(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.side_effect = LookupError
        with pytest.raises(LookupError):
            calibration.get_polynomial_coefficients_from_ccc(store, 'ccc000')

    def test_can_retrive_added_ccc_poly(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = (
            make_calibration(active=True, polynomial=[10, 0, 0, 0, 150, 0])
        )
        assert (
            calibration.get_polynomial_coefficients_from_ccc(store, 'ccc000')
            == [10, 0, 0, 0, 150, 0]
        )

    def test_gets_all_active_cccs(self):
        store = MagicMock(CalibrationStore)
        calibration0 = make_calibration(identifier='ccc0', active=True)
        calibration1 = make_calibration(identifier='ccc1', active=True)
        store.get_all_calibrations.return_value = [
            calibration0, calibration1, make_calibration(active=False)
        ]
        assert (
            calibration.get_active_cccs(store)
            == {'ccc0': calibration0, 'ccc1': calibration1}
        )

    def test_get_polynomial_for_under_construction_raises(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = (
            make_calibration(active=False)
        )
        with pytest.raises(KeyError):
            calibration.get_polynomial_coefficients_from_ccc(store, 'ccc000')


class TestCalculatePolynomial:
    @pytest.mark.parametrize("data_store", (
        calibration.CalibrationData([], [], []),
        calibration.CalibrationData([1], [[1]], [[1]]),
        calibration.CalibrationData([1], [[1]], [[6]]),
        calibration.CalibrationData([1], [[1, 4]], [[6, 2]])
    ))
    def test_too_little_data_raises(self, data_store):

        with pytest.raises(calibration.CCCConstructionError):
            calibration.calculate_polynomial(data_store, 5).tolist()

    def test_calculate_polynomial(self):
        data_store = calibration.CalibrationData(
            source_values=[
                [1, 4, 5],
                [1, 4, 6, 7],
                [3, 8, 11],
                [15, 16],
                [1, 1.5],
                [1.05, 1.9],
            ],
            source_value_counts=[
                [2, 1, 1],
                [1, 3, 1, 2],
                [10, 11, 15],
                [5, 5],
                [103, 121],
                [103, 121],
            ],
            target_value=np.array(
                [151, 615, 8393, 7525, 1694.75, 2327.2024999999994]
            ),
        )
        coeffs = calibration.calculate_polynomial(
            data_store,
            degree=2
        )
        np.testing.assert_allclose(coeffs, [3, 2, 0], rtol=0.001)


class TestGetAllColonyData:

    def test_gets_all_included_colonies_in_empty_ccc(self):
        store = MagicMock(CalibrationStore)
        store.get_measurements_for_calibration.return_value = []
        colonies = calibration.get_all_colony_data(store, 'ccc000')
        assert colonies['source_values'] == []
        assert colonies['source_value_counts'] == []
        assert colonies['target_values'] == []
        assert colonies['min_source_values'] == 0
        assert colonies['max_source_values'] == 0
        assert colonies['max_source_counts'] == 0

    def test_gets_all_included_colonies_in_ccc(self):
        store = MagicMock(CalibrationStore)
        store.get_measurements_for_calibration.return_value = [
            {
                CCCMeasurement.source_values: [1.1],
                CCCMeasurement.source_value_counts: [11],
                CCCMeasurement.cell_count: 500,
            },
            {
                CCCMeasurement.source_values: [2.2, 3.3],
                CCCMeasurement.source_value_counts: [3, 2],
                CCCMeasurement.cell_count: 5000,
            },
            {
                CCCMeasurement.source_values: [4.4, 5.5, 6.6],
                CCCMeasurement.source_value_counts: [6, 5, 4],
                CCCMeasurement.cell_count: 50,
            },
        ]
        colonies = calibration.get_all_colony_data(store, 'ccc000')
        assert colonies == {
            'source_values': [[4.4, 5.5, 6.6], [1.1], [2.2, 3.3]],
            'source_value_counts': [[6, 5, 4], [11], [3, 2]],
            'target_values': [50, 500, 5000],
            'min_source_values': 1.1,
            'max_source_values': 6.6,
            'max_source_counts': 11,
        }


class TestAddCCC:
    def test_valid_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        ccc = make_calibration()
        assert calibration.add_ccc(store, ccc) is True
        store.add_calibration.assert_called_with(ccc)

    def test_none_id(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        ccc = make_calibration(identifier=None)
        assert calibration.add_ccc(store, ccc) is False
        store.add_calibration.assert_not_called()

    def test_duplicate_id(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        ccc = make_calibration()
        assert not calibration.add_ccc(store, ccc)
        store.add_calibration.assert_not_called()


class TestActivateCCC:
    def test_activate(self):
        store = MagicMock(CalibrationStore)
        ccc = make_calibration(
            identifier='ccc001',
            polynomial=[1, 2, 3],
            access_token='password',
            active=False,
        )
        store.get_calibration_by_id.return_value = ccc
        assert (
            calibration.activate_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_called_with(
            'ccc001', CalibrationEntryStatus.Active
        )

    def test_invalid_polynomial(self):
        store = MagicMock(CalibrationStore)
        ccc = make_calibration(
            identifier='ccc001',
            polynomial=None,
            access_token='password',
            active=False,
        )
        store.get_calibration_by_id.return_value = ccc
        assert not (
            calibration.activate_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()

    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.activate_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            polynomial=[1, 2, 3],
            access_token='password',
            active=True,
        )
        assert not (
            calibration.activate_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()


class TestDeleteCCC:
    def test_delete(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=False,
        )
        assert (
            calibration.delete_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_called_with(
            'ccc001', CalibrationEntryStatus.Deleted
        )

    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.delete_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=True,
        )
        assert not (
            calibration.delete_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()


class TestConstructPolynomial:
    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )
        store.set_calibration_polynomial.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=True,
        )
        assert not calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )

        store.set_calibration_status.assert_not_called()

    def test_construct_bad_polynomial(self, badslope_ccc_measurements):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=False,
        )
        store.get_measurements_for_calibration.return_value = [
            {
                CCCMeasurement.source_values: m['source_values'],
                CCCMeasurement.source_value_counts: m['source_value_counts'],
                CCCMeasurement.cell_count: m['cell_count'],
            } for m in badslope_ccc_measurements
        ]
        response = calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )
        assert response['validation'] == 'BadSlope'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])
        assert len(response['calculated_sizes']) == 30
        assert len(response['measured_sizes']) == 30
        assert response['correlation']['p_value'] == pytest.approx(1)
        store.set_calibration_polynomial.assert_not_called()

    def test_construct_good_polynomial(self, full_ccc_measurements):
        store = MagicMock(CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=False,
        )
        measurements = [
            {
                CCCMeasurement.source_values: m['source_values'],
                CCCMeasurement.source_value_counts: m['source_value_counts'],
                CCCMeasurement.cell_count: m['cell_count'],
            } for m in full_ccc_measurements
        ]
        store.get_measurements_for_calibration.return_value = measurements
        response = calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )
        assert response['validation'] == 'OK'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])
        assert len(response['calculated_sizes']) == 62
        assert response['measured_sizes'] == list(sorted(
            measurement[CCCMeasurement.cell_count]
            for measurement in measurements
        ))

        assert response['correlation']['slope'] == pytest.approx(1, abs=0.02)
        assert response['correlation']['intercept'] == pytest.approx(
            71000, rel=0.5
        )
        assert response['correlation']['stderr'] == pytest.approx(
            0.015, rel=0.1
        )
        assert response['correlation']['p_value'] == pytest.approx(0)
        np.testing.assert_allclose(
            response['polynomial_coefficients'],
            [
                5.263000000000004e-05,
                0.004012000000000001,
                0.03962,
                0.9684,
                2.008000000000001e-06,
                0.0,
            ],
            rtol=0.01,
        )
        store.set_calibration_polynomial.assert_called_with('ccc001', {
            CCCPolynomial.coefficients: response['polynomial_coefficients'],
            CCCPolynomial.power: 5,
        })


class TestAddImageToCCC:
    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        image = MagicMock()
        assert not (
            calibration.add_image_to_ccc(
                store, 'ccc001', image, access_token='password',
            )
        )

    def test_already_activated(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            active=True,
            access_token='password',
        )
        image = MagicMock()
        assert not (
            calibration.add_image_to_ccc(
                store, 'ccc001', image, access_token='password',
            )
        )

    def test_add_image(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            active=False,
            access_token='password',
        )
        store.count_images_for_calibration.return_value = 5
        image = MagicMock()
        assert (
            calibration.add_image_to_ccc(
                store, 'ccc001', image, access_token='password',
            )
        )
        image.save.assert_called()
        store.add_image_to_calibration.assert_called_with(
            'ccc001', 'CalibIm_5')


def make_image_metadata(identifier='CalibIm_3'):
    return calibration.get_empty_image_entry(identifier)


class TestGetImageIdentifiersInCCC:
    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.get_image_identifiers_in_ccc(store, 'unknown')
        )

    def test_get_identifiers(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_images_for_calibration.return_value = [
            make_image_metadata(identifier='CalibIm_0'),
            make_image_metadata(identifier='CalibIm_5'),
        ]
        assert (
            calibration.get_image_identifiers_in_ccc(store, 'unknown')
            == ['CalibIm_0', 'CalibIm_5']
        )


class TestSetImageInfo:
    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not calibration.set_image_info(
            store,
            'unknown',
            'CalibIm_2',
            access_token='password',
            fixture='newFixture',
        )
        store.update_calibration_image_with_id.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=True,
            access_token='password',
        )
        assert not calibration.set_image_info(
            store,
            'ccc001',
            'CalibIm_X',
            access_token='password',
            fixture='newFixture',
        )
        store.update_calibration_image_with_id.assert_not_called()

    def test_unknown_image(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = False
        assert not calibration.set_image_info(
            store,
            'ccc001',
            'CalibIm_1',
            access_token='password',
            fixture='newFixture',
        )
        store.update_calibration_image_with_id.assert_not_called()

    def test_update_invalid_info(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = True
        assert not calibration.set_image_info(
            store,
            'ccc001',
            'CalibIm_2',
            access_token='password',
            foobar='new foobar',
        )
        store.update_calibration_image_with_id.assert_not_called()

    def test_update_fixture(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = True
        assert calibration.set_image_info(
            store,
            'ccc001',
            'CalibIm_2',
            access_token='password',
            fixture='newFixture',
        )
        store.update_calibration_image_with_id.assert_called_with(
            'ccc001',
            'CalibIm_2',
            {CCCImage.fixture: 'newFixture'},
        )


class TestSetPlateGridInfo:
    def test_unknown_ccc(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not calibration.set_plate_grid_info(
            store,
            'unknown',
            'CalibIm_2',
            1,
            access_token='password',
            grid_shape=(12, 32),
            grid_cell_size=(100, 200),
        )

    def test_already_activated(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=True,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = True
        store.has_plate_with_id.return_value = True
        assert not calibration.set_plate_grid_info(
            store,
            'unknown',
            'CalibIm_2',
            1,
            access_token='password',
            grid_shape=(12, 32),
            grid_cell_size=(100, 200),
        )

    def test_unknown_image(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = False
        store.has_plate_with_id.return_value = False
        assert not calibration.set_plate_grid_info(
            store,
            'unknown',
            'CalibIm_2',
            1,
            access_token='password',
            grid_shape=(12, 32),
            grid_cell_size=(100, 200),
        )

    def test_add_new_plate(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = True
        store.has_plate_with_id.return_value = False
        store.has_measurements_for_plate.return_value = False
        assert calibration.set_plate_grid_info(
            store,
            'ccc000',
            'CalibIm_2',
            1,
            access_token='password',
            grid_shape=(12, 32),
            grid_cell_size=(100, 200),
        )
        store.add_plate.assert_called_with('ccc000', 'CalibIm_2', 1, {
            CCCPlate.grid_cell_size: (100, 200),
            CCCPlate.grid_shape: (12, 32),
        })
        store.update_plate.assert_not_called()

    def test_update_existing_plate(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = True
        store.has_plate_with_id.return_value = True
        store.has_measurements_for_plate.return_value = False
        assert calibration.set_plate_grid_info(
            store,
            'ccc000',
            'CalibIm_2',
            1,
            access_token='password',
            grid_shape=(12, 32),
            grid_cell_size=(100, 200),
        )
        store.update_plate.assert_called_with('ccc000', 'CalibIm_2', 1, {
            CCCPlate.grid_cell_size: (100, 200),
            CCCPlate.grid_shape: (12, 32),
        })
        store.add_plate.assert_not_called()

    def test_plate_has_measurements(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        store.has_calibration_image_with_id.return_value = True
        store.has_plate_with_id.return_value = True
        store.has_measurements_for_plate.return_value = True
        assert not calibration.set_plate_grid_info(
            store,
            'ccc000',
            'CalibIm_2',
            1,
            access_token='password',
            grid_shape=(12, 32),
            grid_cell_size=(100, 200),
        )
        store.update_plate.assert_not_called()
        store.add_plate.assert_not_called()


class TestGetImageJSONFromCCC:
    def test_unknown_calibration(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
            is None
        )

    def test_unknown_image(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_images_for_calibration.return_value = [
            make_image_metadata(identifier='NotCalibIm0'),
        ]
        assert (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
            is None
        )

    def test_existing_image(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.has_calibration_image_with_id.return_value = True
        image = make_image_metadata(identifier='CalibIm0')
        store.get_images_for_calibration.return_value = [image]
        assert (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
            == image
        )


class TestSetColonyCompressedData:
    def test_unknown_calibration(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
        )

    def test_set_colony(self):
        store = MagicMock(CalibrationStore)
        store.has_calibration_with_id.return_value = True
        calibrationid = 'ccc000'
        store.get_calibration_by_id.return_value = make_calibration(
            identifier=calibrationid,
            active=False,
            access_token='password',
        )
        imageid = 'image0'
        plateid = 'plate0'
        x, y = 0, 0
        image = np.array([
            [0, 1, 1, 1],
            [1, 1, 2, 1],
            [1, 2, 3, 1],
            [1, 1, 1, 9],
        ])
        blob_filter = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=bool)
        background_filter = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ], dtype=bool)
        cell_count = 1234
        assert calibration.set_colony_compressed_data(
            store, 'ccc000', imageid, plateid, x, y, cell_count,
            image, blob_filter, background_filter, access_token='password'
        )
        store.set_measurement.assert_called_with(
            'ccc000', imageid, plateid, x, y, {
                CCCMeasurement.source_values: (0, 1, 2),
                CCCMeasurement.source_value_counts: (1, 2, 1),
                CCCMeasurement.cell_count: 1234,
            },
        )
