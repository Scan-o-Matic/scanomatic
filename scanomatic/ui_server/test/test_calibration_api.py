import os
import json

import pytest
from flask import Flask

from scanomatic.ui_server import calibration_api
from scanomatic.io.paths import Paths
from scanomatic.data_processing import calibration


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
def finalizable_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    _ccc[calibration.CellCountCalibration.deployed_polynomial] = "stiff"
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope='function')
def active_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    _ccc[calibration.CellCountCalibration.deployed_polynomial] = "stiff"
    _ccc[
        calibration.CellCountCalibration.status
    ] = calibration.CalibrationEntryStatus.Active
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope='function')
def deleted_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    _ccc[
        calibration.CellCountCalibration.status
    ] = calibration.CalibrationEntryStatus.Deleted
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope="function")
def test_app():
    app = Flask("Scan-o-Matic UI", template_folder=Paths().ui_templates)
    calibration_api.add_routes(app)
    app.testing = True
    return app.test_client()


class TestFinalizeEndpoint:
    route = "/api/data/calibration/{identifier}/finalize"

    def test_token_not_valid(self, test_app, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'wrongPassword'

        response = test_app.post(
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

    @pytest.mark.parametrize("method", ["get", "put", "delete"])
    def test_finalize_only_supports_post(
            self, method, test_app, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.open(
            self.route.format(identifier=identifier),
            method=method,
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 405
        ), "API call gave unexpected response {} (expected 405)".format(
            response.status)

    def test_finalize_works_when_finished(self, test_app, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 200
        ), "POST gave unexpected response {} (expected 200)".format(
            response.status)

    def test_finalize_fails_when_unfinished(self, test_app, edit_ccc):
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            json.loads(response.data)['reason'] == "Failed to activate ccc"
        ), "POST when unfinished gave unexpected reason {} (expected 'Failed to activate ccc')".format(
            json.loads(response.data)['reason'])
        assert (
            response.status_code == 400
        ), "POST when unfinished gave unexpected response {} (expected 400)".format(
            response.status)

    def test_finalize_fails_when_activated(self, test_app, active_ccc):
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
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

    def test_finalize_fails_when_deleted(self, test_app, deleted_ccc):
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
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
    route = "/api/data/calibration/{identifier}/delete"

    def test_token_not_valid(self, test_app, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'wrongPassword'

        response = test_app.post(
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
            self, method, test_app, finalizable_ccc):
        identifier = finalizable_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.open(
            self.route.format(identifier=identifier),
            method=method,
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 405
        ), "API call gave unexpected response {} (expected 405)".format(
            response.status)

    def test_delete_editable_ccc(self, test_app, edit_ccc):
        identifier = edit_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 200
        ), "POST gave unexpected response {} (expected 200)".format(
            response.status)

    def test_delete_deleted_ccc(self, test_app, deleted_ccc):
        identifier = deleted_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 401
        ), "POST gave unexpected response {} (expected 401)".format(
            response.status)

    def test_delete_active_ccc(self, test_app, active_ccc):
        identifier = active_ccc[
            calibration.CellCountCalibration.identifier]
        token = 'password'

        response = test_app.post(
            self.route.format(identifier=identifier),
            data={"access_token": token},
            follow_redirects=True
        )
        assert (
            response.status_code == 401
        ), "POST gave unexpected response {} (expected 401)".format(
            response.status)
