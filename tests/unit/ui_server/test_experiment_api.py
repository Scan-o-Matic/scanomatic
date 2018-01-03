from StringIO import StringIO
from datetime import timedelta
from hashlib import sha256
import httplib as HTTPStatus
import json
import os
import pytest
from types import MethodType
from flask import Flask, url_for
from mock import MagicMock, Mock, patch
import pytest

from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.io.scanstore import ScanStore, UnknownProjectError
from scanomatic.ui_server import experiment_api


@pytest.fixture
def scanstore():
    return MagicMock(ScanStore)


@pytest.fixture
def rpc_client():
    return MagicMock()


@pytest.fixture
def app(scanstore, rpc_client):
    app = Flask(__name__, template_folder=Paths().ui_templates)
    app.config['scanstore'] = scanstore
    experiment_api.add_routes(app, rpc_client)
    return app


@pytest.fixture(scope="function")
def test_app(app, rpc_client):
    def _post_json(self, uri, data, **kwargs):
        return self.post(
            uri,
            data=json.dumps(data),
            content_type='application/json',
            **kwargs
        )
    test_app = app.test_client()
    test_app.post_json = MethodType(_post_json, test_app)
    test_app.rpc_client = rpc_client
    return test_app


class TestFeatureExtractEndpoint:

    route = '/api/project/feature_extract'

    @staticmethod
    def jailed_path(path):
        return os.path.abspath(
            path.replace('root', Config().paths.projects_root, 1))

    def test_no_action(self, test_app):
        response = test_app.post(
            self.route,
            data={},
            follow_redirects=True
        )
        assert response.status_code != 200

    @patch(
        'scanomatic.ui_server.experiment_api.FeaturesFactory._validate_analysis_directory',
        return_value=True)
    def test_keep_previous_qc(self, validator_mock, test_app):

        test_app.rpc_client.create_feature_extract_job.return_value = 'Hi'

        response = test_app.post_json(
            self.route,
            {
                'analysis_directory': 'root/test',
                'keep_qc': 1,
            },
            follow_redirects=True
        )

        assert test_app.rpc_client.create_feature_extract_job.called
        assert test_app.rpc_client.create_feature_extract_job.called_with({
            'try_keep_qc': True,
            'analysis_directory': self.jailed_path('root/test/')
        })
        assert response.status_code == 200

    @patch(
        'scanomatic.ui_server.experiment_api.FeaturesFactory._validate_analysis_directory',
        return_value=True)
    def test_dont_keep_previous_qc(self, validator_mock, test_app):

        test_app.rpc_client.create_feature_extract_job.return_value = 'Hi'

        response = test_app.post_json(
            self.route,
            {
                'analysis_directory': 'root/test',
                'keep_qc': 0,
            },
            follow_redirects=True
        )

        assert test_app.rpc_client.create_feature_extract_job.called
        assert test_app.rpc_client.create_feature_extract_job.called_with({
            'try_keep_qc': False,
            'analysis_directory': self.jailed_path('root/test/')
        })
        assert response.status_code == 200


class TestPostScan:
    @pytest.fixture
    def data(self):
        return {
            'project': 'my/project',
            'scan_index': 3,
            'timedelta': 60,
            'image': (StringIO('I am an image'), 'image.tiff'),
            'digest': sha256('I am an image').hexdigest(),
        }

    @pytest.fixture
    def url(self):
        return url_for('scans')

    def test_post_scan(self, client, url, data, scanstore):
        res = client.post(url, data=data)
        assert res.status_code == HTTPStatus.CREATED, res.data
        scanstore.add_scan.assert_called()
        project, scan = scanstore.add_scan.call_args[0]
        assert project == 'my/project'
        assert scan.index == 3
        assert scan.timedelta == timedelta(seconds=60)
        assert scan.image.read() == 'I am an image'

    @pytest.mark.parametrize('key, value', [
        ('project', ''),
        ('project', '..'),
        ('project', 'foo/'),
        ('project', '/foo'),
        ('scan_index', 'xxx'),
        ('scan_index', -1),
        ('timedelta', 'xxx'),
        ('timedelta', -1),
        ('image', 'zzz'),
        ('digest', 1234),
        ('digest', 'zzzz'),
        ('image', (StringIO('I am an image'), 'image.xls')),
    ])
    def test_invalid_data(self, client, url, data, key, value):
        data[key] = value
        res = client.post(url, data=data)
        assert res.status_code == HTTPStatus.BAD_REQUEST

    @pytest.mark.parametrize('key', [
        'project', 'scan_index', 'timedelta', 'image', 'image', 'digest',
    ])
    def test_missing_data(self, client, url, data, key):
        del data[key]
        res = client.post(url, data=data)
        assert res.status_code == HTTPStatus.BAD_REQUEST

    def test_unknown_project(self, client, url, data, scanstore):
        scanstore.add_scan.side_effect = UnknownProjectError()
        res = client.post(url, data=data)
        assert res.status_code == HTTPStatus.BAD_REQUEST
