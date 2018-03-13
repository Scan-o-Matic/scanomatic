from __future__ import absolute_import

from itertools import product
import json
import os

from flask import Flask
import mock
import numpy as np
import pytest
from scipy.ndimage import center_of_mass
from scipy.stats import norm

from scanomatic.data_processing import calibration
from scanomatic.io.ccc_data import (
    parse_ccc, CellCountCalibration, CalibrationEntryStatus, CCCPolynomial
)
from scanomatic.io.paths import Paths
from scanomatic.models.analysis_model import COMPARTMENTS
from scanomatic.ui_server import calibration_api
from scanomatic.ui_server.calibration_api import (
    get_bounding_box_for_colony, get_colony_detection
)
import scanomatic.ui_server.database as db


def _fixture_load_ccc(app, rel_path):
    parent = os.path.dirname(__file__)
    with open(os.path.join(parent, rel_path), 'rb') as fh:
        data = json.load(fh)
    _ccc = parse_ccc(data)
    if _ccc:
        with app.app_context():
            store = db.getcalibrationstore()
            store.add_calibration(_ccc)
        return _ccc
    raise ValueError("The `{0}` is not valid/doesn't parse".format(rel_path))


@pytest.fixture(scope='function')
def edit_ccc(app):
    _ccc = _fixture_load_ccc(app, 'data/test_good.ccc')
    with app.app_context():
        store = db.getcalibrationstore()
        store.set_calibration_polynomial(
            _ccc[CellCountCalibration.identifier], None)
    return _ccc


@pytest.fixture(scope='function')
def finalizable_ccc(app):
    _ccc = _fixture_load_ccc(app, 'data/test_good.ccc')
    yield _ccc


@pytest.fixture(scope='function')
def active_ccc(app):
    _ccc = _fixture_load_ccc(app, 'data/test_good.ccc')
    with app.app_context():
        store = db.getcalibrationstore()
        store.set_calibration_status(
            _ccc[CellCountCalibration.identifier],
            calibration.CalibrationEntryStatus.Active)
    return _ccc


@pytest.fixture(scope='function')
def deleted_ccc(app):
    _ccc = _fixture_load_ccc(app, 'data/test_good.ccc')
    with app.app_context():
        store = db.getcalibrationstore()
        store.set_calibration_status(
            _ccc[CellCountCalibration.identifier],
            calibration.CalibrationEntryStatus.Deleted)
    return _ccc


class TestFinalizeEndpoint:
    route = "/calibration/{identifier}/finalize"

    def test_token_not_valid(self, client, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'wrongPassword'

        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        print(response.data)
        expected = "Invalid access token or CCC not under construction"
        assert (
            json.loads(response.data)['reason'] == expected
        ), "POST with bad token gave wrong reason {} (expected '{}')".format(
            json.loads(response.data)['reason'], expected)
        assert (
            response.status_code == 401
        ), "POST with bad token gave wrong response {} (expected 401)".format(
            response.status)

    @pytest.mark.parametrize("method", ["get", "put", "delete"])
    def test_finalize_only_supports_post(
            self, method, client, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = client.open(
            self.route.format(identifier=identifier),
            method=method,
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 405
        ), "API call gave unexpected response {} (expected 405)".format(
            response.status)

    def test_finalize_works_when_finished(self, client, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'
        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 200
        ), "POST gave unexpected response {} (expected 200)".format(
            response.status)

    def test_finalize_fails_when_unfinished(self, client, edit_ccc):
        identifier = edit_ccc[calibration.CellCountCalibration.identifier]
        token = 'password'
        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            json.loads(response.data)['reason'] == "Failed to activate ccc"
        ), "POST when unfinished gave wrong reason {} (expected 'Failed to activate ccc')".format(
            json.loads(response.data)['reason'])
        assert (
            response.status_code == 400
        ), "POST when unfinished gave wrong response {} (expected 400)".format(
            response.status)

    def test_finalize_fails_when_activated(self, client, active_ccc):
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        expected = "Invalid access token or CCC not under construction"
        assert (
            json.loads(response.data)['reason'] == expected
        ), "POST with bad token gave unexpected reason {} (expected '{}')".format(
            json.loads(response.data)['reason'], expected)
        assert (
            response.status_code == 401
        ), "POST with bad token gave unexpected response {} (expected 401)".format(
            response.status)

    def test_finalize_fails_when_deleted(self, client, deleted_ccc):
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        expected = "Invalid access token or CCC not under construction"
        assert (
            json.loads(response.data)['reason'] == expected
        ), "POST with bad token gave unexpected reason {} (expected '{}')".format(
            json.loads(response.data)['reason'], expected)
        assert (
            response.status_code == 401
        ), "POST with bad token gave unexpected response {} (expected 401)".format(
            response.status)


class TestDeleteEndpoint:
    route = "/calibration/{identifier}/delete"

    def test_token_not_valid(self, client, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'wrongPassword'

        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        expected = "Invalid access token or CCC not under construction"
        assert (
            json.loads(response.data)['reason'] == expected
        ), "POST with bad token gave unexpected reason {0} (expected '{1}')".format(
            json.loads(response.data)['reason'], expected)
        assert (
             response.status_code == 401
        ), "POST with bad token gave unexpected response {} (expected 401)".format(
            response.status)

    @pytest.mark.parametrize("method", ["get", "put", "delete"])
    def test_delete_only_supports_post(
            self, method, client, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = client.open(
            self.route.format(identifier=identifier),
            method=method,
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 405
        ), "API call gave unexpected response {} (expected 405)".format(
            response.status)

    def test_delete_editable_ccc(self, client, edit_ccc):
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'
        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 200
        ), "POST gave unexpected response {} (expected 200)".format(
            response.status)

    def test_delete_deleted_ccc(self, client, deleted_ccc):
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 401
        ), "POST gave unexpected response {} (expected 401)".format(
            response.status)

    def test_delete_active_ccc(self, client, active_ccc):
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = client.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 401
        ), "POST gave unexpected response {} (expected 401)".format(
            response.status)


class TestCompressCalibration:
    url = "/calibration/ccc0/image/img0/plate/0/compress/colony/0/0"

    @pytest.fixture(autouse=True)
    def get_image_json_from_ccc(self):
        with mock.patch(
            'scanomatic.ui_server.calibration_api.calibration.get_image_json_from_ccc',
            return_value={},
        ):
            yield

    @pytest.fixture
    def set_colony_compressed_data(self):
        with mock.patch(
            'scanomatic.ui_server.calibration_api.calibration.set_colony_compressed_data'
        ) as function:
            yield function

    @pytest.fixture
    def set_colony_compressed_data(self):
        with mock.patch(
            'scanomatic.ui_server.calibration_api.calibration.set_colony_compressed_data'
        ) as function:
            yield function

    @pytest.fixture
    def params(self):
        return {
            "blob": [[0] * 20, [1] * 20],
            'background': [[1] * 20, [0] * 20],
            "cell_count": 42,
            'access_token': 'XXX'
        }

    def test_valid_params(self, client, set_colony_compressed_data, params):
        response = client.post(self.url, data=json.dumps(params))
        assert response.status_code == 200
        args, kwargs = set_colony_compressed_data.call_args
        assert args == (mock.ANY, 'ccc0', 'img0', 0, 0, 0, 42)
        assert kwargs['access_token'] == 'XXX'
        assert np.array_equal(
            kwargs['background_filter'],
            np.array([[True] * 20, [False] * 20])
        )
        assert np.array_equal(
            kwargs['blob_filter'],
            np.array([[False] * 20, [True] * 20]),
        )
        assert kwargs['access_token'] == 'XXX'

    def test_missing_cell_count(
            self, client, set_colony_compressed_data, params):
        del params['cell_count']
        response = client.post(self.url, data=json.dumps(params))
        assert response.status_code == 400
        assert (
            json.loads(response.data)['reason']
            == 'Missing expected parameter cell_count'
        )
        set_colony_compressed_data.assert_not_called()

    def test_non_integer_cell_count(
            self, client, set_colony_compressed_data, params):
        params['cell_count'] = 'abc'
        response = client.post(self.url, data=json.dumps(params))
        assert response.status_code == 400
        assert (
            json.loads(response.data)['reason']
            == 'cell_count should be an integer'
        )
        set_colony_compressed_data.assert_not_called()

    def test_negative_cell_count(
            self, client, set_colony_compressed_data, params):
        params['cell_count'] = -1
        response = client.post(self.url, data=json.dumps(params))
        assert response.status_code == 400
        assert (
            json.loads(response.data)['reason']
            == 'cell_count should be greater or equal than zero'
        )
        set_colony_compressed_data.assert_not_called()


class TestConstructCalibration:

    url = '/calibration/{ccc}/construct/{power}'

    @pytest.mark.parametrize(
        'ccc_identifier,power,access_token,expected_status',
        (
            ('XXX', 5, 'heelo', 401),  # Unknown ccc
            ('testgoodedit', 5, 'heelo', 401),  # Bad access_token
            ('testgoodedit', -1,  'password', 404),  # Bad power
        )
    )
    def test_fails_with_bad_parameters(
        self, client, ccc_identifier, power, access_token, expected_status,
        finalizable_ccc
    ):
        # finalizable_ccc is needed for enpoint to have that ccc loaded
        response = client.post(
            self.url.format(ccc=ccc_identifier, power=power),
            data=json.dumps({'acccess_token': access_token}))
        assert response.status_code == expected_status

    # TODO
    @pytest.mark.skip
    def test_returns_a_polynomial(self, client, finalizable_ccc):
        ccc_identifier = 'testgoodedit'
        power = 5
        access_token = 'password'
        response = client.post(
            self.url.format(ccc=ccc_identifier, power=power),
            data=json.dumps({'access_token': access_token}))
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['polynomial_coefficients']) == power + 1
        assert data['validation'] == 'OK'
        assert (
            len(data['measured_sizes']) == len(data['calculated_sizes'])
        )
        assert all(
            key in data['colonies'] for key in
            (
                'source_values', 'source_value_counts', 'target_values',
                'min_source_values', 'max_source_values',
                'max_source_counts',
            )
        )

    def test_returns_fail_reason(self, client, finalizable_ccc):
        with mock.patch(
            'scanomatic.data_processing.calibration.construct_polynomial',
            return_value={
                'polynomial_coefficients': [2, 1, 0],
                'measured_sizes': [0],
                'calculated_sizes': [0],
                'validation': 'BadSlope',
                'correlation': {
                    'slope': 5,
                    'intercept': 3,
                    'p_value': 0.01,
                    'stderr': 0.5,
                },
            }
        ) as construct_polynomial:

            ccc_identifier = 'testgoodedit'
            power = 2
            access_token = 'password'
            response = client.post(
                self.url.format(ccc=ccc_identifier, power=power),
                data=json.dumps({'access_token': access_token}))

            assert response.status_code == 400
            construct_polynomial.assert_called()

            data = json.loads(response.data)
            assert data['reason'].startswith(
                u"Construction refused. "
                "Validation of polynomial says: BadSlope "
                "(y = 1.00E+00 x^1 + 2.00E+00 x^2) "
                "correlation: {"
            )
            assert "'stderr': 0.5" in data['reason']
            assert "'p_value': 0.01" in data['reason']
            assert "'intercept': 3" in data['reason']
            assert "'slope': 5" in data['reason']
