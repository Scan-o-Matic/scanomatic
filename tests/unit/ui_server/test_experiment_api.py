from __future__ import absolute_import

from StringIO import StringIO
from datetime import timedelta
from hashlib import sha256
import httplib as HTTPStatus
import json
import os
from types import MethodType

from flask import Flask, url_for
from mock import MagicMock, patch
import pytest

from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.ui_server import experiment_api, ui_server


@pytest.fixture
def rpc_client():
    return MagicMock()


@pytest.fixture
def app(rpc_client):
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(ui_server.__file__), 'templates',
        )
    )
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
