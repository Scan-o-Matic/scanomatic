import os
import json
from collections import namedtuple
import mock
import tempfile

import numpy as np
import pytest

from scanomatic.data_processing import calibration
from scanomatic.io.paths import Paths

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


def test_expand_data_lenghts(ccc):
    data = calibration._collect_all_included_data(ccc)
    counts = data.source_value_counts
    exp_vals, _, _, _ = calibration._get_expanded_data(data)
    assert all(np.sum(c) == len(v) for c, v in zip(counts, exp_vals))


def test_expand_data_sums(ccc):

    data = calibration._collect_all_included_data(ccc)
    counts = data.source_value_counts
    values = data.source_values
    data_sums = np.array(
        tuple(np.sum(np.array(c) * np.array(v))
              for c, v in zip(counts, values)))

    exp_vals, _, _, _ = calibration._get_expanded_data(data)
    expanded_sums = np.array(tuple(v.sum() for v in exp_vals), dtype=np.float)
    np.testing.assert_allclose(expanded_sums, data_sums)


def test_expand_data_targets(ccc):

    data = calibration._collect_all_included_data(ccc)
    _, targets, _, _ = calibration._get_expanded_data(data)
    np.testing.assert_allclose(
        targets.astype(np.float),
        data.target_value)


def test_expand_vector_length():

    counts = [20, 3, 5, 77, 2, 35]
    values = [20, 21, 23, 24, 26, 27]
    expanded = calibration._expand_compressed_vector(values, counts, np.float)
    assert sum(counts) == expanded.size


def test_expanded_vector_sum():
    counts = [20, 3, 5, 77, 2, 35]
    values = [20, 21, 23, 24, 26, 27]
    data_sum = np.sum(np.array(counts) * np.array(values))
    expanded = calibration._expand_compressed_vector(values, counts, np.float)
    np.testing.assert_allclose(expanded.sum(), data_sum)


def test_calibration_opt_func():

    poly = calibration.get_calibration_optimization_function(2)
    assert poly([2], 1, 1)[0] == 6
    assert poly([2], 2, 0)[0] == 4
    assert poly([2], 0, 2)[0] == 8
    poly = calibration.get_calibration_optimization_function(4)
    assert poly([1], 1, 1)[0] == 2
    assert poly([1], 2, 0)[0] == 2
    assert poly([1], 0, 2)[0] == 2
    assert poly([2], 0, 1)[0] == 16


def test_get_im_slice():
    """Test that _get_im_slice handles floats"""
    image = np.arange(0, 42).reshape((6, 7))
    model_tuple = namedtuple("Model", ['x1', 'x2', 'y1', 'y2'])
    model = model_tuple(1.5, 3.5, 2.5, 4.5)
    assert calibration._get_im_slice(image, model).sum() == 207


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
    _ccc = calibration._parse_ccc(data)
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

    def test_validate_bad_correlation(self, data_store_bad_ccc):

        poly = np.poly1d([2, 1])
        assert (
            calibration.validate_polynomial(data_store_bad_ccc, poly) is
            calibration.CalibrationValidation.BadSlope
        )

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

    def test_ccc_calculate_polynomial(self, data_store_bad_ccc):

        poly_coeffs = calibration.calculate_polynomial(data_store_bad_ccc, 5)
        assert len(poly_coeffs) == 6
        assert poly_coeffs[0] != 0
        assert poly_coeffs[-2] != 0
        assert poly_coeffs[-1] == 0
        assert all(v == 0 for v in poly_coeffs[1:4])

    def test_construct_bad_polynomial(self, edit_bad_slope_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_bad_slope_ccc[
            calibration.CellCountCalibration.identifier]
        poly_name = 'test'
        power = 5
        token = 'password'

        response = calibration.construct_polynomial(
            identifier, poly_name, power, access_token=token)

        assert (
            response['validation'] is
            calibration.CalibrationValidation.BadSlope)

    @pytest.mark.skip("Unreliable with current data")
    def test_construct_good_polynomial(self, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_ccc[calibration.CellCountCalibration.identifier]
        poly_name = 'test'
        power = 5
        token = 'password'
        print(identifier)

        response = calibration.construct_polynomial(
            identifier, poly_name, power, access_token=token)

        assert (
            response['validation'] is
            calibration.CalibrationValidation.OK)

        assert response['polynomial_name'] == poly_name
        assert response['polynomial_degree'] == power
        assert response['ccc'] == identifier

        assert len(response['calculated_sizes']) == 16
        assert len(response['measured_sizes']) == 16


class TestActivateCCC:

    @pytest.mark.parametrize("polynomial", [
        {
            calibration.CCCPolynomial.power: "apa",
            calibration.CCCPolynomial.coefficients: [0, 1]
        },
        {
            calibration.CCCPolynomial.power: 1.0,
            calibration.CCCPolynomial.coefficients: [0, 1]
        },
        {
            calibration.CCCPolynomial.power: 2,
            calibration.CCCPolynomial.coefficients: [0, 1]
        },
        {"browser": 2, "coffee": [0, 1]},
        {'power': 1, 'coefficients': [0, 1]},
    ])
    def test_polynomial_malformed(self, polynomial):
        with pytest.raises(calibration.ActivationError):
            calibration.validate_polynomial_struct(polynomial)

    @pytest.mark.parametrize("polynomial", [
        {
            calibration.CCCPolynomial.power: 0,
            calibration.CCCPolynomial.coefficients: [0]
        },
        {
            calibration.CCCPolynomial.power: 1,
            calibration.CCCPolynomial.coefficients: [0, 1]
        },
        {
            calibration.CCCPolynomial.power: 2,
            calibration.CCCPolynomial.coefficients: [1, 2, 3]
        },
    ])
    def test_polynomial_correct(self, polynomial):
        assert calibration.validate_polynomial_struct(polynomial) is None

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

        poly_name = 'test'
        power = 5

        response = calibration.construct_polynomial(
            identifier, poly_name, power, access_token=token)

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
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')
    def test_save_ccc(self, save_mock, validator_mock, ccc):

        assert calibration.save_ccc_to_disk(ccc)
        assert not validator_mock.called
        assert save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')
    def test_add_existing_ccc(self, save_mock, validator_mock, ccc):

        assert not calibration.add_ccc(ccc)
        assert not validator_mock.called
        assert not save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')
    def test_add_ccc(self, save_mock, validator_mock):

        ccc = calibration.get_empty_ccc('Bogus schmogus', 'Dr Lus')
        assert calibration.add_ccc(ccc)
        assert not validator_mock.called
        assert save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')
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
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')
    def test_delete_ccc(self, save_mock, validator_mock, ccc):

        assert calibration.delete_ccc(
            ccc[calibration.CellCountCalibration.identifier],
            access_token='not used, but needed')
        assert validator_mock.called
        assert save_mock.called

    @mock.patch(
        'scanomatic.data_processing.calibration._ccc_edit_validator',
        return_value=True)
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')
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
    @mock.patch('scanomatic.data_processing.calibration._save_ccc_to_disk')

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
        assert measurement[calibration.CCCMeasurement.source_values] == (0, 1, 2)

    def test_source_value_counts(self, measurement):
        assert measurement[calibration.CCCMeasurement.source_value_counts] == (1, 2, 1)

    def test_cell_count(self, measurement):
        assert measurement[calibration.CCCMeasurement.cell_count] == 1234
