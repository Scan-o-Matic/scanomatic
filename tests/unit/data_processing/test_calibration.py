import os
import json
from collections import namedtuple
import mock
import tempfile

import numpy as np
import pytest

from scanomatic.data_processing import calibration
from scanomatic.io.paths import Paths
from scanomatic.io import ccc_data

# The curve fitting is done with coefficients as e^x, some tests
# want to have that expression to be zero while still real.
# Then this constant is used.
EVAL_AS_ZERO = -99

# Needs adding for all ccc_* paths if tests extended
# Don't worry Paths is a singleton
Paths().ccc_folder = os.path.join(tempfile.gettempdir(), 'tempCCC')
Paths().ccc_file_pattern = os.path.join(
    Paths().ccc_folder, '{0}.ccc')


@pytest.fixture(scope='function')
def ccc():

    _ccc = calibration.get_empty_ccc('test-ccc', 'pytest')
    calibration.__CCC[_ccc[calibration.CellCountCalibration.identifier]] = _ccc
    yield _ccc
    calibration.reload_cccs()


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

    def test_invalid_token(self, ccc):

        assert not calibration.is_valid_edit_request(
            ccc[calibration.CellCountCalibration.identifier]
        ), "Edit request worked, despite missing token"

        assert not calibration.is_valid_edit_request(
            ccc[calibration.CellCountCalibration.identifier],
            access_token='bad'
        ), "Edit request worked, despite bad token"

    def test_valid_token(self, ccc):

        assert calibration.is_valid_edit_request(
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token]
        ) is True, "Edit request failed, despite valid token."


def _fixture_load_ccc(rel_path):
    parent = os.path.dirname(__file__)
    with open(os.path.join(parent, rel_path), 'rb') as fh:
        data = json.load(fh)
    _ccc = ccc_data.parse_ccc(data)
    if _ccc:
        calibration.__CCC[
            _ccc[calibration.CellCountCalibration.identifier]] = _ccc
        return _ccc
    raise ValueError("The `{0}` is not valid/doesn't parse".format(rel_path))


@pytest.fixture(scope='function')
def edit_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    yield _ccc
    calibration.reload_cccs()


@pytest.fixture(scope='function')
def edit_bad_slope_ccc():
    _ccc = _fixture_load_ccc('data/test_badslope.ccc')
    yield _ccc
    calibration.reload_cccs()


@pytest.fixture(scope='function')
def full_ccc():
    _ccc = _fixture_load_ccc('data/TESTUMz.ccc')
    yield _ccc
    calibration.reload_cccs()

@pytest.fixture(scope='function')
def finalizable_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    yield _ccc
    calibration.reload_cccs()


@pytest.fixture(scope='function')
def active_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    _ccc[
        calibration.CellCountCalibration.status
    ] = calibration.CalibrationEntryStatus.Active
    yield _ccc
    calibration.reload_cccs()


@pytest.fixture(scope='function')
def deleted_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    _ccc[
        calibration.CellCountCalibration.status
    ] = calibration.CalibrationEntryStatus.Deleted
    yield _ccc
    calibration.reload_cccs()


@pytest.fixture(scope='function')
def data_store_bad_ccc(edit_bad_slope_ccc):
    return calibration._collect_all_included_data(edit_bad_slope_ccc)


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

    def test_construct_bad_polynomial(self, edit_bad_slope_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_bad_slope_ccc[
            calibration.CellCountCalibration.identifier]
        power = 5
        token = 'password'

        response = calibration.construct_polynomial(
            identifier, power, access_token=token)

        assert response['validation'] != 'OK'
        assert all(coeff >= 0 for coeff in response['polynomial_coefficients'])

        assert len(response['calculated_sizes']) == 30
        assert len(response['measured_sizes']) == 30

        assert response['correlation']['p_value'] == pytest.approx(1)

    def test_construct_good_polynomial(self, full_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = full_ccc[calibration.CellCountCalibration.identifier]
        power = 5
        token = full_ccc[calibration.CellCountCalibration.edit_access_token]
        full_ccc[calibration.CellCountCalibration.status] = \
            calibration.CalibrationEntryStatus.UnderConstruction

        response = calibration.construct_polynomial(
            identifier, power, access_token=token)

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

    def test_activated_status_is_set(self, finalizable_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'
        assert (
            finalizable_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC not initialized with UnderConstruction entry status"

        calibration.activate_ccc(identifier, access_token=token)

        assert (
            finalizable_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC activation failed"

    def test_activated_ccc_not_editable(self, active_ccc):
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
            identifier, power, access_token=token)

        assert response is None, "Could edit active CCC but shouldn't have"

    def test_activated_status_is_not_set_if_no_poly(self, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC not initialized with UnderConstruction entry status"

        edit_ccc[calibration.CellCountCalibration.polynomial] = None
        calibration.activate_ccc(identifier, access_token=token)

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC activation worked but shouldn't have"

    def test_activate_active_ccc(self, active_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC not initialized with Active entry status"

        status = calibration.delete_ccc(identifier, access_token=token)

        assert (
            status is None
        ), "CCC activation returned unexcepted status {}".format(status)

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC activation worked, but shouldn't have"

    def test_activate_deleted_ccc(self, deleted_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC not initialized with Deleted entry status"

        status = calibration.delete_ccc(identifier, access_token=token)

        assert (
            status is None
        ), "CCC activation returned {} (expected: {})".format(status, None)

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC activation had unforseen consequences"


class TestDeleteCCC:

    def test_delete_active_ccc(self, active_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC not initialized with Active entry status"

        status = calibration.delete_ccc(identifier, access_token=token)

        assert (
            status is None
        ), "CCC deletion returned unexcepted status {}".format(status)

        assert (
            active_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Active
        ), "CCC deletion worked, but shouldn't have"

    def test_delete_deleted_ccc(self, deleted_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC not initialized with Deleted entry status"

        status = calibration.delete_ccc(identifier, access_token=token)

        assert (
            status is None
        ), "CCC deletion returned {} (expected: {})".format(status, None)

        assert (
            deleted_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC deletion had unforseen consequences"

    def test_delete_editable_ccc(self, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC has wrong status"

        status = calibration.delete_ccc(identifier, access_token=token)

        assert (
            status is True
        ), "CCC deletion returned {} (expected: {})".format(status, True)

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.Deleted
        ), "CCC deletion didn't work but it should have"


class TestGettingActiveCCCs:

    @mock.patch(
        'scanomatic.data_processing.calibration.save_ccc_to_disk',
        return_value=True)
    def setup_method(self, func, my_mock):

        calibration.reload_cccs()

        ccc1 = calibration.get_empty_ccc('Cylon', 'Boomer')
        self._ccc_id1 = ccc1[calibration.CellCountCalibration.identifier]
        ccc1[calibration.CellCountCalibration.polynomial] = {
            calibration.CCCPolynomial.power: 5,
            calibration.CCCPolynomial.coefficients: [10, 0, 0, 0, 150, 0]
        }
        ccc1[calibration.CellCountCalibration.status] = \
            calibration.CalibrationEntryStatus.Active
        calibration.add_ccc(ccc1)

        ccc2 = calibration.get_empty_ccc('Deep Ones', 'Stross')
        self._ccc_id2 = ccc2[calibration.CellCountCalibration.identifier]
        ccc2[calibration.CellCountCalibration.polynomial] = {
            calibration.CCCPolynomial.power: 5,
            calibration.CCCPolynomial.coefficients: [10, 0, 0, 0, 150, 0]
        }
        calibration.add_ccc(ccc2)

    def teardown_method(self):
        calibration.reload_cccs()

    def test_exists_default_ccc_polynomial(self):

        assert calibration.get_polynomial_coefficients_from_ccc('default')

    @pytest.mark.parametrize('ccc_identifier', [None, 'TheDoctor', 'DEEPON'])
    def test_invalid_ccc_raises_exception_for_poly(self, ccc_identifier):

        with pytest.raises(KeyError):

            calibration.get_polynomial_coefficients_from_ccc(ccc_identifier)

    def test_can_retrive_added_ccc_poly(self):

        assert calibration.get_polynomial_coefficients_from_ccc(self._ccc_id1)

    def test_gets_all_active_cccs(self):

        assert len(calibration.get_active_cccs()) == 2

    def test_get_polynomial_for_under_construction_raises(self):

        with pytest.raises(KeyError):

            calibration.get_polynomial_coefficients_from_ccc(self._ccc_id2)


class TestSaving:

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    def test_save_ccc(self, save_mock, validator_mock, ccc):

        assert calibration.save_ccc_to_disk(ccc)
        assert not validator_mock.called
        assert save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    def test_add_existing_ccc(self, save_mock, validator_mock, ccc):

        assert not calibration.add_ccc(ccc)
        assert not validator_mock.called
        assert not save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    def test_add_ccc(self, save_mock, validator_mock):

        ccc = calibration.get_empty_ccc('Bogus schmogus', 'Dr Lus')
        assert calibration.add_ccc(ccc)
        assert not validator_mock.called
        assert save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    @mock.patch(
        'scanomatic.data_processing.calibration.has_valid_polynomial',
        return_value=True)
    def test_activate_ccc(
            self, poly_validator_mock, save_mock, validator_mock, ccc):

        assert calibration.activate_ccc(
            ccc[calibration.CellCountCalibration.identifier],
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_mock.called
        assert poly_validator_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    def test_delete_ccc(self, save_mock, validator_mock, ccc):

        assert calibration.delete_ccc(
            ccc[calibration.CellCountCalibration.identifier],
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    def test_add_image_to_ccc(self, save_mock, validator_mock, ccc):

        image_mock = mock.Mock()
        assert calibration.add_image_to_ccc(
            ccc[calibration.CellCountCalibration.identifier], image_mock,
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_mock.called
        assert image_mock.save.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration.save_ccc')
    def test_add_image_to_ccc(self, save_mock, validator_mock, ccc):
        assert calibration.set_image_info(
            ccc[calibration.CellCountCalibration.identifier], 0,
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_mock.called


class TestCCCEditValidator:

    def test_allowed_knowing_id(self, ccc):

        assert calibration._ccc_edit_validator(
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token])

    def test_bad_id_existing_ccc(self, ccc):

        assert not calibration._ccc_edit_validator(
            ccc[calibration.CellCountCalibration.identifier],
            access_token='Something is wrong')

    def test_non_existing_ccc(self):

        ccc = calibration.get_empty_ccc('Bogus schmogus leoii', 'Dr Lus')
        assert not calibration._ccc_edit_validator(
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token])

    def test_no_access_(self, ccc):

        assert not calibration._ccc_edit_validator(
                ccc[calibration.CellCountCalibration.identifier])


class TestSetColonyCompressedData:

    @pytest.fixture(autouse=True)
    def save_ccc_to_disk(self):
        with mock.patch(
            'scanomatic.data_processing.calibration.save_ccc_to_disk',
            return_value=True
        ):
            yield

    @pytest.fixture
    def measurement(self, ccc):
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
            identifier, image_identifier, plate_id, x, y, cell_count,
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

    def test_gets_all_included_colonies_in_empty_ccc(self, ccc):
        ccc_id = ccc[calibration.CellCountCalibration.identifier]
        colonies = calibration.get_all_colony_data(ccc_id)
        assert colonies['source_values'] == []
        assert colonies['source_value_counts'] == []
        assert colonies['target_values'] == []
        assert colonies['min_source_values'] == 0
        assert colonies['max_source_values'] == 0
        assert colonies['max_source_counts'] == 0

    def test_gets_all_included_colonies_in_ccc(self, edit_ccc):
        ccc_id = edit_ccc[calibration.CellCountCalibration.identifier]
        colonies = calibration.get_all_colony_data(ccc_id)
        assert len(colonies['source_values']) == 16
        assert len(colonies['source_value_counts']) == 16
        assert len(colonies['target_values']) == 16
        assert colonies['min_source_values'] == 0.21744831955999988
        assert colonies['max_source_values'] == 30.68582517095
        assert colonies['max_source_counts'] == 21562

    def test_doesnt_scramble_data_while_sorting(self, edit_ccc):
        ccc_id = edit_ccc[calibration.CellCountCalibration.identifier]
        colonies = calibration.get_all_colony_data(ccc_id)
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
