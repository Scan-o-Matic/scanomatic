import os
import json
from collections import namedtuple

import numpy as np
import pytest

from scanomatic.data_processing import calibration

data = calibration.load_data_file()


@pytest.fixture(scope='module')
def ccc():

    _ccc = calibration.get_empty_ccc('test-ccc', 'pytest')
    calibration.__CCC[_ccc[calibration.CellCountCalibration.identifier]] = _ccc
    yield _ccc
    del calibration.__CCC[_ccc[calibration.CellCountCalibration.identifier]]


def test_load_data():

    assert calibration.load_data_file() is not None


def test_load_calibration():

    assert calibration.load_calibration() is not None


def test_expand_data_lenghts():

    counts = data.source_value_counts
    exp_vals, _, _, _ = calibration._get_expanded_data(data)
    assert all(np.sum(c) == len(v) for c, v in zip(counts, exp_vals))


def test_expand_data_sums():

    counts = data.source_value_counts
    values = data.source_values
    data_sums = np.array(
        tuple(np.sum(np.array(c) * np.array(v))
              for c, v in zip(counts, values)))

    exp_vals, _, _, _ = calibration._get_expanded_data(data)
    expanded_sums = np.array(tuple(v.sum() for v in exp_vals), dtype=np.float)
    np.testing.assert_allclose(expanded_sums, data_sums)


def test_expand_data_targets():

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

        assert not calibration.is_valid_token(
            ccc[calibration.CellCountCalibration.identifier])

        assert not calibration.is_valid_token(
            ccc[calibration.CellCountCalibration.identifier],
            access_token='bad')

    def test_valid_token(self, ccc):

        assert calibration.is_valid_token(
            ccc[calibration.CellCountCalibration.identifier],
            access_token=ccc[
                calibration.CellCountCalibration.edit_access_token]) is True


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
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope='function')
def edit_bad_slope_ccc():
    _ccc = _fixture_load_ccc('data/test_badslope.ccc')
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope='function')
def finalizable_ccc():
    _ccc = _fixture_load_ccc('data/test_finalizable.ccc')
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


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

        response = calibration.constuct_polynomial(
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

        response = calibration.constuct_polynomial(
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
        {"power": "apa", "coefficients": [0, 1]},
        {"power": 1.0, "coefficients": [0, 1]},
        {"power": 2, "coefficients": [0, 1]},
        {"browser": 2, "coffee": [0, 1]},
    ])
    def test_polynomial_malformed(self, polynomial):
        with pytest.raises(calibration.ActivationError):
            calibration.validate_polynomial_struct(polynomial)

    @pytest.mark.parametrize("polynomial", [
        {"power": 0, "coefficients": [0]},
        {"power": 1, "coefficients": [0, 1]},
        {"power": 2, "coefficients": [1, 2, 3]},
    ])
    def test_polynomial_correct(self, polynomial):
        assert calibration.validate_polynomial_struct(polynomial) is None

    def test_no_has_selected_polynomial(self, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        with pytest.raises(calibration.ActivationError):
            calibration.has_valid_polynomial(edit_ccc)

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

    def test_activated_ccc_not_editable(self, finalizable_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            finalizable_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC not initialized with UnderConstruction entry status"

        calibration.activate_ccc(identifier, access_token=token)

        poly_name = 'test'
        power = 5

        response = calibration.constuct_polynomial(
            identifier, poly_name, power, access_token=token)

        assert response is None

    def test_activated_status_is_not_set(self, edit_ccc):
        # The fixture needs to be included, otherwise test is not correct
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC not initialized with UnderConstruction entry status"

        calibration.activate_ccc(identifier, access_token=token)

        assert (
            edit_ccc[calibration.CellCountCalibration.status] ==
            calibration.CalibrationEntryStatus.UnderConstruction
        ), "CCC activation worked but shouldn't have"
