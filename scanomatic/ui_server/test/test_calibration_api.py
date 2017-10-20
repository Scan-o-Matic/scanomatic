import os
import json

import pytest
import numpy as np
from flask import Flask
from itertools import product

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


@pytest.mark.parametrize('data,expected', (
    (None, None),
    ([], None),
    ((1, 2), (1, 2)),
    (('1', '2'), (1, 2)),
))
def test_get_int_tuple(data, expected):

    assert calibration_api.get_int_tuple(data) == expected


def test_get_bounding_box_for_colony():

    # 3x3 colony grid
    grid = np.array(
        [
            # Colony positions' y according to their positions in the grid
            [
                [51, 102, 151],
                [51, 101, 151],
                [50, 102, 152],
            ],

            # X according to their positions on the grid
            [
                [75, 125, 175],
                [75, 123, 175],
                [75, 125, 175],
            ]
        ]
    )
    width = 50
    height = 30

    for x, y in product(range(3), range(3)):

        box = calibration_api.get_bounding_box_for_colony(
            grid, x, y, width, height)

        assert (box['center'] == grid[:, y, x]).all()
        assert box['yhigh'] - box['ylow'] == height + 1
        assert box['xhigh'] - box['xlow'] == width + 1
        assert box['xlow'] >= 0
        assert box['ylow'] >= 0


def test_get_boundin_box_for_colony_if_grid_partially_outside():
    """only important that never gets negative numbers for box"""

    grid = np.array(
        [
            [
                [-5, 10],
                [51, 101],
            ],

            [
                [10, 125],
                [-5, 123],
            ]
        ]
    )
    width = 50
    height = 30

    for x, y in product(range(2), range(2)):

        box = calibration_api.get_bounding_box_for_colony(
            grid, x, y, width, height)

        assert box['center'][0] >= 0
        assert box['center'][1] >= 0
        assert box['xlow'] >= 0
        assert box['ylow'] >= 0


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
