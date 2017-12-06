from __future__ import absolute_import
import os
import json

import pytest
import mock
import numpy as np
from scipy.stats import norm
from scipy.ndimage import center_of_mass
from flask import Flask
from itertools import product

from scanomatic.models.analysis_model import COMPARTMENTS
from scanomatic.ui_server import calibration_api
from scanomatic.io.paths import Paths
from scanomatic.io.ccc_data import parse_ccc
from scanomatic.data_processing import calibration
from scanomatic.ui_server.calibration_api import (
    get_bounding_box_for_colony, get_colony_detection
)


def _fixture_load_ccc(rel_path):
    parent = os.path.dirname(__file__)
    with open(os.path.join(parent, rel_path), 'rb') as fh:
        data = json.load(fh)
    _ccc = parse_ccc(data)
    if _ccc:
        calibration.__CCC[
            _ccc[calibration.CellCountCalibration.identifier]] = _ccc
        return _ccc
    raise ValueError("The `{0}` is not valid/doesn't parse".format(rel_path))


@pytest.fixture(scope='function')
def edit_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    _ccc[calibration.CellCountCalibration.polynomial] = None
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope='function')
def finalizable_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
    yield _ccc
    calibration.__CCC.pop(_ccc[calibration.CellCountCalibration.identifier])


@pytest.fixture(scope='function')
def active_ccc():
    _ccc = _fixture_load_ccc('data/test_good.ccc')
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
    app.register_blueprint(calibration_api.blueprint)
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
    route = "/{identifier}/finalize"

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
        ), "POST with bad token gave wrong reason {} (expected '{}')".format(
            json.loads(response.data)['reason'], expected)
        assert (
            response.status_code == 401
        ), "POST with bad token gave wrong response {} (expected 401)".format(
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

        with mock.patch(
                'scanomatic.data_processing.calibration.save_ccc') as save_ccc:
            response = test_app.post(
                self.route.format(identifier=identifier),
                data={"access_token": token},
                follow_redirects=True
            )
            save_ccc.assert_called()

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
        ), "POST when unfinished gave wrong reason {} (expected 'Failed to activate ccc')".format(
            json.loads(response.data)['reason'])
        assert (
            response.status_code == 400
        ), "POST when unfinished gave wrong response {} (expected 400)".format(
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
    route = "/{identifier}/delete"

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

        with mock.patch(
                'scanomatic.data_processing.calibration.save_ccc') as save_ccc:
            response = test_app.post(
                self.route.format(identifier=identifier),
                data={"access_token": token},
                follow_redirects=True
            )
            save_ccc.assert_called()
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


class TestCompressCalibration:
    url = "/ccc0/image/img0/plate/0/compress/colony/0/0"

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

    def test_valid_params(self, test_app, set_colony_compressed_data, params):
        response = test_app.post(self.url, data=json.dumps(params))
        assert response.status_code == 200
        args, kwargs = set_colony_compressed_data.call_args
        assert args == ('ccc0', 'img0', 0, 0, 0, 42)
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
            self, test_app, set_colony_compressed_data, params):
        del params['cell_count']
        response = test_app.post(self.url, data=json.dumps(params))
        assert response.status_code == 400
        assert (
            json.loads(response.data)['reason']
            == 'Missing expected parameter cell_count'
        )
        set_colony_compressed_data.assert_not_called()

    def test_non_integer_cell_count(
            self, test_app, set_colony_compressed_data, params):
        params['cell_count'] = 'abc'
        response = test_app.post(self.url, data=json.dumps(params))
        assert response.status_code == 400
        assert (
            json.loads(response.data)['reason']
            == 'cell_count should be an integer'
        )
        set_colony_compressed_data.assert_not_called()

    def test_negative_cell_count(
            self, test_app, set_colony_compressed_data, params):
        params['cell_count'] = -1
        response = test_app.post(self.url, data=json.dumps(params))
        assert response.status_code == 400
        assert (
            json.loads(response.data)['reason']
            == 'cell_count should be greater or equal than zero'
        )
        set_colony_compressed_data.assert_not_called()


class TestConstructCalibration:

    url = '/{ccc}/construct/{power}'

    @pytest.mark.parametrize(
        'ccc_identifier,power,access_token,expected_status',
        (
            ('XXX', 5, 'heelo', 401),  # Unknown ccc
            ('testgoodedit', 5, 'heelo', 401),  # Bad access_token
            ('testgoodedit', -1,  'password', 404),  # Bad power
        )
    )
    def test_fails_with_bad_parameters(
        self, test_app, ccc_identifier, power, access_token, expected_status,
        finalizable_ccc
    ):
        # finalizable_ccc is needed for enpoint to have that ccc loaded
        response = test_app.post(
            self.url.format(ccc=ccc_identifier, power=power),
            data=json.dumps({'acccess_token': access_token}))
        assert response.status_code == expected_status

    def test_returns_a_polynomial(self, test_app, finalizable_ccc):
        with mock.patch(
                'scanomatic.data_processing.calibration.save_ccc') as save_ccc:
            ccc_identifier = 'testgoodedit'
            power = 5
            access_token = 'password'
            response = test_app.post(
                self.url.format(ccc=ccc_identifier, power=power),
                data=json.dumps({'access_token': access_token}))

            assert response.status_code == 200
            save_ccc.assert_called()

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

    def test_returns_fail_reason(self, test_app, finalizable_ccc):
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
            response = test_app.post(
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


@pytest.mark.parametrize('grid,x,y,w,h,expected', (
    (np.array([[[10]], [[10]]]), 0, 0, 5, 6,
     {'ylow': 7, 'yhigh': 14, 'xlow': 8, 'xhigh': 13, 'center': (10, 10)}),
    (np.array([[[10, 20]], [[10, 10]]]), 1, 0, 5, 6,
     {'ylow': 17, 'yhigh': 24, 'xlow': 8, 'xhigh': 13, 'center': (20, 10)}),
    (np.array([[[5]], [[10]]]), 0, 0, 10, 20,
     {'ylow': 0, 'yhigh': 16, 'xlow': 5, 'xhigh': 16, 'center': (5, 10)}),
))
def test_bounding_box_for_colony(grid, x, y, w, h, expected):
    result = get_bounding_box_for_colony(grid, x, y, w, h)
    assert result == expected


@pytest.fixture(scope='function')
def colony_image():
    im = np.ones((25, 25)) * 80
    cell_vector = norm.pdf(np.arange(-5, 6)/2.)
    colony = np.multiply.outer(cell_vector, cell_vector) * 20
    im[6:17, 5:16] -= colony
    return im


class TestGetColonyDetection:

    def test_colony_is_darker(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        background = grid_cell.get_item(COMPARTMENTS.Background).filter_array
        assert (
            grid_cell.source[blob].mean() <
            grid_cell.source[background].mean()
        )

    def test_blob_and_background_dont_overlap(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        background = grid_cell.get_item(COMPARTMENTS.Background).filter_array
        assert (blob & background).sum() == 0

    def test_blob_is_of_expected_size(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        assert blob.sum() == pytest.approx(100, abs=10)

    def test_blob_has_expected_center(self, colony_image):
        grid_cell = get_colony_detection(colony_image)
        blob = grid_cell.get_item(COMPARTMENTS.Blob).filter_array
        assert center_of_mass(blob) == pytest.approx((11, 10), abs=1)
