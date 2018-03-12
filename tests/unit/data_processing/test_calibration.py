from __future__ import absolute_import

from collections import namedtuple
import json
import os
import tempfile

import mock
from mock import MagicMock, Mock
import numpy as np
import pytest

from scanomatic.data_processing import calibration
from scanomatic.io import ccc_data
from scanomatic.io.ccc_data import (
    CalibrationEntryStatus, CCCImage, CCCMeasurement, CCCPlate, CCCPolynomial,
    CellCountCalibration
)
from scanomatic.io.paths import Paths

# The curve fitting is done with coefficients as e^x, some tests
# want to have that expression to be zero while still real.
# Then this constant is used.
EVAL_AS_ZERO = -99

# Needs adding for all ccc_* paths if tests extended
# Don't worry Paths is a singleton
Paths().ccc_folder = os.path.join(tempfile.gettempdir(), 'tempCCC')
Paths().ccc_file_pattern = os.path.join(
    Paths().ccc_folder, '{0}.ccc')


@pytest.fixture(autouse=True)
def save_ccc():
    def fakesave(ccc):
        print("Saving ccc {} (not realy)".format(
            ccc[calibration.CellCountCalibration.identifier]
        ))
        return True
    with mock.patch(
        'scanomatic.data_processing.calibration.save_ccc',
        side_effect=fakesave,
    ) as save_ccc:
        yield save_ccc


@pytest.fixture
def store():
    return calibration.CalibrationStore()


@pytest.fixture(scope='function')
def ccc(store):
    _ccc = calibration.get_empty_ccc(store, 'test-ccc', 'pytest')
    store.add_calibration(_ccc)
    return _ccc


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


class TestAccessToken:

    def test_missing_token(self, store, ccc):
        assert not calibration.is_valid_edit_request(
            store,
            ccc[calibration.CellCountCalibration.identifier]
        ), "Edit request worked, despite missing token"

    def test_bad_token(self, store, ccc):
        assert not calibration.is_valid_edit_request(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token='bad'
        ), "Edit request worked, despite bad token"

    def test_valid_token(self, store, ccc):
        assert calibration.is_valid_edit_request(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token]
        ) is True, "Edit request failed, despite valid token."


def _fixture_load_ccc(store, rel_path):
    parent = os.path.dirname(__file__)
    with open(os.path.join(parent, rel_path), 'rb') as fh:
        data = json.load(fh)
    _ccc = ccc_data.parse_ccc(data)
    if _ccc:
        store.add_calibration(_ccc)
        return _ccc
    raise ValueError("The `{0}` is not valid/doesn't parse".format(rel_path))


@pytest.fixture(scope='function')
def edit_ccc(store):
    _ccc = _fixture_load_ccc(store, 'data/test_good.ccc')
    return _ccc


@pytest.fixture(scope='function')
def edit_bad_slope_ccc(store):
    _ccc = _fixture_load_ccc(store, 'data/test_badslope.ccc')
    return _ccc


@pytest.fixture(scope='function')
def full_ccc(store):
    _ccc = _fixture_load_ccc(store, 'data/TESTUMz.ccc')
    return _ccc


@pytest.fixture(scope='function')
def finalizable_ccc(store):
    _ccc = _fixture_load_ccc(store, 'data/test_good.ccc')
    return _ccc


@pytest.fixture(scope='function')
def active_ccc(store):
    _ccc = _fixture_load_ccc(store, 'data/test_good.ccc')
    _ccc[
        calibration.CellCountCalibration.status
    ] = calibration.CalibrationEntryStatus.Active
    return _ccc


@pytest.fixture(scope='function')
def deleted_ccc(store):
    _ccc = _fixture_load_ccc(store, 'data/test_good.ccc')
    _ccc[
        calibration.CellCountCalibration.status
    ] = calibration.CalibrationEntryStatus.Deleted
    return _ccc


@pytest.fixture(scope='function')
def data_store_bad_ccc(store, edit_bad_slope_ccc):
    return calibration._collect_all_included_data(
        store, edit_bad_slope_ccc[CellCountCalibration.identifier]
    )


class TestEditCCC:

    @pytest.mark.parametrize('slope,p_value,stderr,expected', (
        (0.99, 0.01, 0.001, calibration.CalibrationValidation.OK),
        (0.89, 0.01, 0.001, calibration.CalibrationValidation.BadSlope),
        (0.99, 0.5, 0.001, calibration.CalibrationValidation.BadStatistics),
        (0.99, 0.01, 0.06, calibration.CalibrationValidation.BadStatistics),
    ))
    def test_validate_bad_correlation(self, slope, p_value, stderr, expected):

        assert calibration.validate_polynomial(
            slope, p_value, stderr) == expected

    def test_ccc_is_in_edit_mode(self, edit_ccc):

        assert (
            edit_ccc[calibration.CellCountCalibration.status] is
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "Not edit mode"

        assert (
            edit_ccc[calibration.CellCountCalibration.identifier]
        ), "Missing Identifier"

    def test_ccc_collect_all_data_has_equal_length(self, data_store_bad_ccc):

        lens = [
            len(getattr(data_store_bad_ccc, k)) for k in
            ('target_value', 'source_values', 'source_value_counts')
        ]
        assert all(v == lens[0] for v in lens)

    def test_ccc_collect_all_data_entries_has_equal_length(
            self, data_store_bad_ccc):

        values = data_store_bad_ccc.source_values
        counts = data_store_bad_ccc.source_value_counts
        assert all(len(v) == len(c) for v, c in zip(values, counts))

    def test_construct_bad_polynomial(self, edit_bad_slope_ccc, store):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_bad_slope_ccc[
            calibration.CellCountCalibration.identifier]
        power = 5
        token = 'password'

        response = calibration.construct_polynomial(
            store, identifier, power, access_token=token)

        assert response['validation'] != 'OK'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])

        assert len(response['calculated_sizes']) == 30
        assert len(response['measured_sizes']) == 30

        assert response['correlation']['p_value'] == pytest.approx(1)

    def test_construct_good_polynomial(self, full_ccc, store):
        # The fixture needs to be included, otherwise test is not correct
        identifier = full_ccc[calibration.CellCountCalibration.identifier]
        power = 5
        token = full_ccc[calibration.CellCountCalibration.edit_access_token]
        full_ccc[calibration.CellCountCalibration.status] = \
            calibration.CalibrationEntryStatus.UnderConstruction

        response = calibration.construct_polynomial(
            store, identifier, power, access_token=token)

        assert response['validation'] == 'OK'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])

        assert len(response['calculated_sizes']) == 62
        assert response['measured_sizes'] == [
            640000,
            680000,
            800000,
            1160000,
            1320000,
            1660000,
            1900000,
            2060000,
            2240000,
            2280000,
            2360000,
            3120000,
            3240000,
            3260000,
            4100000,
            4240000,
            4460000,
            4580000,
            5120000,
            5380000,
            5520000,
            5760000,
            6600000,
            7980000,
            8240000,
            9200000,
            9840000,
            10300000,
            10380000,
            10760000,
            10800000,
            10960000,
            11040000,
            11580000,
            11920000,
            12120000,
            12400000,
            13160000,
            13860000,
            14100000,
            14160000,
            14220000,
            14220000,
            14440000,
            14720000,
            14940000,
            15120000,
            15160000,
            15380000,
            15660000,
            15920000,
            16180000,
            16440000,
            16700000,
            16700000,
            16820000,
            17060000,
            17200000,
            17700000,
            17900000,
            18240000,
            18340000,
        ]

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


class TestActivateCCC:
    def test_has_selected_polynomial(self, finalizable_ccc):
        # The fixture needs to be included, otherwise test is not correct
        assert (
            calibration.has_valid_polynomial(finalizable_ccc) is None
        ), "CCC does not have valid polynomial"

    def test_activated_status_is_set(self, store, finalizable_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'
        assert (
            finalizable_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC not initialized with UnderConstruction entry status"

        calibration.activate_ccc(store, identifier, access_token=token)

        assert (
            finalizable_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC activation failed"

    def test_activated_ccc_not_editable(self, store, active_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC not initialized with UnderConstruction entry status"

        power = 5

        response = calibration.construct_polynomial(
            store, identifier, power, access_token=token)

        assert response is None, "Could edit active CCC but shouldn't have"

    def test_activated_status_is_not_set_if_no_poly(self, store, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC not initialized with UnderConstruction entry status"

        edit_ccc[calibration.CellCountCalibration.polynomial] = None
        calibration.activate_ccc(store, identifier, access_token=token)

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC activation worked but shouldn't have"

    def test_activate_active_ccc(self, store, active_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC not initialized with Active entry status"

        status = calibration.delete_ccc(store, identifier, access_token=token)

        assert (
            status is None
        ), "CCC activation returned unexcepted status {}".format(status)

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC activation worked, but shouldn't have"

    def test_activate_deleted_ccc(self, store, deleted_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC not initialized with Deleted entry status"

        status = calibration.delete_ccc(store, identifier, access_token=token)

        assert (
            status is None
        ), "CCC activation returned {} (expected: {})".format(status, None)

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC activation had unforseen consequences"


class TestDeleteCCC:

    def test_delete_active_ccc(self, store, active_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC not initialized with Active entry status"

        status = calibration.delete_ccc(store, identifier, access_token=token)

        assert (
            status is None
        ), "CCC deletion returned unexcepted status {}".format(status)

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC deletion worked, but shouldn't have"

    def test_delete_deleted_ccc(self, store, deleted_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC not initialized with Deleted entry status"

        status = calibration.delete_ccc(store, identifier, access_token=token)

        assert (
            status is None
        ), "CCC deletion returned {} (expected: {})".format(status, None)

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC deletion had unforseen consequences"

    def test_delete_editable_ccc(self, store, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC has wrong status"

        status = calibration.delete_ccc(store, identifier, access_token=token)

        assert (
            status is True
        ), "CCC deletion returned {} (expected: {})".format(status, True)

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC deletion didn't work but it should have"


class TestGettingActiveCCCs:

    @pytest.fixture(scope='function', autouse=True)
    def setup(self, store):
        ccc1 = calibration.get_empty_ccc(store, 'Cylon', 'Boomer')
        self._ccc_id1 = ccc1[calibration.CellCountCalibration.identifier]
        ccc1[calibration.CellCountCalibration.polynomial] = {
            calibration.CCCPolynomial.power: 5,
            calibration.CCCPolynomial.coefficients: [10, 0, 0, 0, 150, 0]
        }
        ccc1[calibration.CellCountCalibration.status] = \
            calibration.CalibrationEntryStatus.Active
        calibration.add_ccc(store, ccc1)

        ccc2 = calibration.get_empty_ccc(store, 'Deep Ones', 'Stross')
        self._ccc_id2 = ccc2[calibration.CellCountCalibration.identifier]
        ccc2[calibration.CellCountCalibration.polynomial] = {
            calibration.CCCPolynomial.power: 5,
            calibration.CCCPolynomial.coefficients: [10, 0, 0, 0, 150, 0]
        }
        calibration.add_ccc(store, ccc2)

    def test_exists_default_ccc_polynomial(self, store):

        assert calibration.get_polynomial_coefficients_from_ccc(
            store, 'default')

    @pytest.mark.parametrize('ccc_identifier', [None, 'TheDoctor', 'DEEPON'])
    def test_invalid_ccc_raises_exception_for_poly(
        self, store, ccc_identifier
    ):
        with pytest.raises(KeyError):
            calibration.get_polynomial_coefficients_from_ccc(
                store, ccc_identifier)

    def test_can_retrive_added_ccc_poly(self, store):
        assert calibration.get_polynomial_coefficients_from_ccc(
            store, self._ccc_id1)

    def test_gets_all_active_cccs(self, store):
        assert len(calibration.get_active_cccs(store)) == 2

    def test_get_polynomial_for_under_construction_raises(self, store):
        with pytest.raises(KeyError):
            calibration.get_polynomial_coefficients_from_ccc(
                store, self._ccc_id2)


class TestSaving:

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_save_ccc(self, validator_mock, save_ccc, store, ccc):
        save_ccc.reset_mock()
        assert calibration.save_ccc_to_disk(store, ccc)
        assert not validator_mock.called
        assert save_ccc.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_add_existing_ccc(self, validator_mock, save_ccc, store, ccc):
        save_ccc.reset_mock()
        assert not calibration.add_ccc(store, ccc)
        assert not validator_mock.called
        assert not save_ccc.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_add_ccc(self, validator_mock, save_ccc, store):
        save_ccc.reset_mock()
        ccc = calibration.get_empty_ccc(store, 'Bogus schmogus', 'Dr Lus')
        assert calibration.add_ccc(store, ccc)
        assert not validator_mock.called
        assert save_ccc.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch(
        'scanomatic.data_processing.calibration.has_valid_polynomial',
        return_value=True)
    def test_activate_ccc(
            self, poly_validator_mock, validator_mock, save_ccc, store, ccc):
        save_ccc.reset_mock()
        assert calibration.activate_ccc(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_ccc.called
        assert poly_validator_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_delete_ccc(self, validator_mock, save_ccc, store, ccc):
        save_ccc.reset_mock()
        assert calibration.delete_ccc(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_ccc.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_add_image_to_ccc(self, validator_mock, save_ccc, store, ccc):
        save_ccc.reset_mock()
        image_mock = mock.Mock()
        assert calibration.add_image_to_ccc(
            store,
            ccc[calibration.CellCountCalibration.identifier], image_mock,
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_ccc.called
        assert image_mock.save.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_set_image_info(self, validator_mock, save_ccc, store, full_ccc):
        save_ccc.reset_mock()
        assert calibration.set_image_info(
            store,
            full_ccc[CellCountCalibration.identifier],
            full_ccc[CellCountCalibration.images][0][CCCImage.identifier],
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_ccc.called


class TestCCCEditValidator:

    def test_allowed_knowing_id(self, store, ccc):
        assert calibration._ccc_edit_validator(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token])

    def test_bad_id_existing_ccc(self, store, ccc):
        assert not calibration._ccc_edit_validator(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token='Something is wrong')

    def test_non_existing_ccc(self, store):
        ccc = calibration.get_empty_ccc(
            store, 'Bogus schmogus leoii', 'Dr Lus')
        assert not calibration._ccc_edit_validator(
            store,
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token])

    def test_no_access_(self, store, ccc):
        assert not calibration._ccc_edit_validator(
            store,
            ccc[calibration.CellCountCalibration.identifier])


class TestSetColonyCompressedData:
    @pytest.fixture
    def measurement(self, store, ccc):
        identifier = ccc[calibration.CellCountCalibration.identifier]
        access_token = ccc[calibration.CellCountCalibration.edit_access_token]
        image_identifier = 'image0'
        plate_id = 'plate0'
        colony_data = {}
        ccc[calibration.CellCountCalibration.images] = [
            {
                calibration.CCCImage.identifier: image_identifier,
                calibration.CCCImage.plates: {
                    plate_id: {
                        calibration.CCCPlate.compressed_ccc_data:
                            {(0, 0): colony_data},
                    }
                }
            }
        ]
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

        calibration.set_colony_compressed_data(
            store, identifier, image_identifier, plate_id, x, y, cell_count,
            image, blob_filter, background_filter, access_token=access_token
        )

        return (
            ccc[calibration.CellCountCalibration.images][0]
            [calibration.CCCImage.plates]['plate0']
            [calibration.CCCPlate.compressed_ccc_data][(0, 0)]
        )

    def test_source_values(self, measurement):
        assert measurement[
            calibration.CCCMeasurement.source_values] == (0, 1, 2)

    def test_source_value_counts(self, measurement):
        assert measurement[
            calibration.CCCMeasurement.source_value_counts] == (1, 2, 1)

    def test_cell_count(self, measurement):
        assert measurement[calibration.CCCMeasurement.cell_count] == 1234


class TestConstructPolynomial:

    @pytest.fixture(scope='session')
    def poly2(self):
        return calibration.get_calibration_optimization_function(2)

    @pytest.fixture(scope='session')
    def poly4(self):
        return calibration.get_calibration_optimization_function(4)

    @pytest.mark.parametrize("data_store", (
        calibration.CalibrationData([], [], []),
        calibration.CalibrationData([1], [[1]], [[1]]),
        calibration.CalibrationData([1], [[1]], [[6]]),
        calibration.CalibrationData([1], [[1, 4]], [[6, 2]])
    ))
    def test_too_little_data_raises(self, data_store):

        with pytest.raises(calibration.CCCConstructionError):
            calibration.calculate_polynomial(data_store, 5).tolist()

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

    def test_gets_all_included_colonies_in_empty_ccc(self, store, ccc):
        ccc_id = ccc[calibration.CellCountCalibration.identifier]
        colonies = calibration.get_all_colony_data(store, ccc_id)
        assert colonies['source_values'] == []
        assert colonies['source_value_counts'] == []
        assert colonies['target_values'] == []
        assert colonies['min_source_values'] == 0
        assert colonies['max_source_values'] == 0
        assert colonies['max_source_counts'] == 0

    def test_gets_all_included_colonies_in_ccc(self, store, edit_ccc):
        ccc_id = edit_ccc[calibration.CellCountCalibration.identifier]
        colonies = calibration.get_all_colony_data(store, ccc_id)
        assert len(colonies['source_values']) == 16
        assert len(colonies['source_value_counts']) == 16
        assert len(colonies['target_values']) == 16
        assert colonies['min_source_values'] == 0.21744831955999988
        assert colonies['max_source_values'] == 30.68582517095
        assert colonies['max_source_counts'] == 21562

    def test_doesnt_scramble_data_while_sorting(self, store, edit_ccc):
        ccc_id = edit_ccc[calibration.CellCountCalibration.identifier]
        colonies = calibration.get_all_colony_data(store, ccc_id)
        assert colonies['source_values'][0] == [
            0.79925410930999963, 1.7992541093099996, 2.7992541093099996,
            3.7992541093099996, 4.7992541093099996,
        ]
        assert colonies['source_value_counts'][0] == [
            5491, 134, 28, 10, 2,
        ]
        assert colonies['target_values'][0] == 60000.0
        assert colonies['source_values'][-1] == [
            4.6858251709500003, 5.6858251709500003, 6.6858251709500003,
            7.6858251709500003, 8.6858251709500003, 9.6858251709500003,
            10.68582517095, 11.68582517095, 12.68582517095, 13.68582517095,
            14.68582517095, 15.68582517095, 16.68582517095, 17.68582517095,
            18.68582517095, 19.68582517095, 20.68582517095, 21.68582517095,
            22.68582517095, 23.68582517095, 24.68582517095, 25.68582517095,
            26.68582517095, 27.68582517095, 28.68582517095, 29.68582517095,
            30.68582517095,
        ]
        assert colonies['source_value_counts'][-1] == [
            1, 7, 50, 96, 81, 54, 74, 78, 56, 67, 82, 70, 94, 61, 82, 95, 94,
            116, 348, 1265, 2900, 4655, 9718, 1982, 2171, 770, 1,
        ]
        assert colonies['target_values'][-1] == 42000000.0


def make_calibration(
    identifier='ccc000',
    polynomial=None,
    access_token='password',
    active=False,
):
    ccc = ccc_data.get_empty_ccc_entry(identifier, 'Bogus schmogus', 'Dr Lus')
    if polynomial is not None:
        ccc[CellCountCalibration.polynomial] = (
            ccc_data.get_polynomal_entry(len(polynomial) - 1, polynomial)
        )
    ccc[CellCountCalibration.edit_access_token] = access_token
    if active:
        ccc[CellCountCalibration.status] = CalibrationEntryStatus.Active
    else:
        ccc[CellCountCalibration.status] = (
            CalibrationEntryStatus.UnderConstruction
        )
    return ccc


class TestAddCCC:
    def test_valid_ccc(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        ccc = make_calibration()
        assert calibration.add_ccc(store, ccc) is True
        store.add_calibration.assert_called_with(ccc)

    def test_none_id(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        ccc = make_calibration(identifier=None)
        assert calibration.add_ccc(store, ccc) is False
        store.add_calibration.assert_not_called()

    def test_duplicate_id(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = True
        ccc = make_calibration()
        assert not calibration.add_ccc(store, ccc)
        store.add_calibration.assert_not_called()


class TestActivateCCC:
    def test_activate(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.activate_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.delete_ccc(store, 'ccc001', access_token='password')
        )
        store.set_calibration_status.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )
        store.set_calibration_polynomial.assert_not_called()

    def test_already_activated(self):
        store = MagicMock(calibration.CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=True,
        )
        assert not calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )

        store.set_calibration_status.assert_not_called()

    @pytest.fixture
    def bad_measurements(self, edit_bad_slope_ccc):
        return [
            measurement
            for image in edit_bad_slope_ccc[CellCountCalibration.images]
            for plate in image[CCCImage.plates].values()
            for measurement in plate[CCCPlate.compressed_ccc_data].values()
        ]

    def test_construct_bad_polynomial(self, bad_measurements):
        store = MagicMock(calibration.CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=False,
        )
        store.get_measurements_for_calibration.return_value = bad_measurements
        response = calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )
        assert response['validation'] == 'BadSlope'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])
        assert len(response['calculated_sizes']) == 30
        assert len(response['measured_sizes']) == 30
        assert response['correlation']['p_value'] == pytest.approx(1)
        store.set_calibration_polynomial.assert_not_called()

    @pytest.fixture
    def good_measurements(self, full_ccc):
        return [
            measurement
            for image in full_ccc[CellCountCalibration.images]
            for plate in image[CCCImage.plates].values()
            for measurement in plate[CCCPlate.compressed_ccc_data].values()
        ]

    def test_construct_good_polynomial(self, good_measurements):
        store = MagicMock(calibration.CalibrationStore)
        store.get_calibration_by_id.return_value = make_calibration(
            identifier='ccc001',
            access_token='password',
            active=False,
        )
        store.get_measurements_for_calibration.return_value = good_measurements
        response = calibration.construct_polynomial(
            store, 'ccc001', power=5, access_token='password',
        )
        assert response['validation'] == 'OK'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])
        assert len(response['calculated_sizes']) == 62
        assert response['measured_sizes'] == list(sorted(
            measurement[CCCMeasurement.cell_count]
            for measurement in good_measurements
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
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        image = MagicMock()
        assert not (
            calibration.add_image_to_ccc(
                store, 'ccc001', image, access_token='password',
            )
        )

    def test_already_activated(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store.add_image_to_calibration.assert_called_with('ccc001', {
            CCCImage.identifier: 'CalibIm_5',
            CCCImage.plates: {},
            CCCImage.grayscale_name: None,
            CCCImage.grayscale_source_values: None,
            CCCImage.grayscale_target_values: None,
            CCCImage.marker_x: None,
            CCCImage.marker_y: None,
            CCCImage.fixture: None,
        })


    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    def test_add_image_to_ccc(self, validator_mock, save_ccc, store, ccc):
        save_ccc.reset_mock()
        image_mock = mock.Mock()
        assert calibration.add_image_to_ccc(
            store,
            ccc[calibration.CellCountCalibration.identifier], image_mock,
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_ccc.called
        assert image_mock.save.called


def make_image_metadata(identifier='CalibIm_3'):
    return calibration.get_empty_image_entry(identifier)


class TestGetImageIdentifiersInCCC:
    def test_unknown_ccc(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.get_image_identifiers_in_ccc(store, 'unknown')
        )

    def test_get_identifiers(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(spec=calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
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
            CCCPlate.compressed_ccc_data: {},
        })
        store.update_plate_with_id.assert_not_called()

    def test_update_existing_plate(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store.update_plate_with_id.assert_called_with('ccc000', 'CalibIm_2', 1, {
            CCCPlate.grid_cell_size: (100, 200),
            CCCPlate.grid_shape: (12, 32),
        })
        store.add_plate.assert_not_called()

    def test_plate_has_measurements(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_calibration_by_id.return_value = make_calibration(
            active=False,
            access_token='password',
        )
        img = make_image_metadata(identifier='CalibIm_2')
        img[CCCImage.plates][1] = {
            CCCPlate.grid_cell_size: None,
            CCCPlate.grid_shape: None,
            CCCPlate.compressed_ccc_data: True
        }
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
        store.update_plate_with_id.assert_not_called()
        store.add_plate.assert_not_called()


class TestGetImageJSONFromCCC:
    def test_unknown_calibration(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
            is None
        )

    def test_unknown_image(self):
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = True
        store.get_images_for_calibration.return_value = [
            make_image_metadata(identifier='NotCalibIm0'),
        ]
        assert (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
            is None
        )

    def test_existing_image(self):
        store = MagicMock(calibration.CalibrationStore)
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
        store = MagicMock(calibration.CalibrationStore)
        store.has_calibration_with_id.return_value = False
        assert not (
            calibration.get_image_json_from_ccc(store, 'ccc000', 'CalibIm0')
        )

    def test_set_colony(self):
        store = MagicMock(calibration.CalibrationStore)
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
