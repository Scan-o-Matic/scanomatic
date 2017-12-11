import pytest
import os
import logging
import json

from flask import Flask
from mock import Mock, patch

from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config
from scanomatic.ui_server import experiment_api


@pytest.fixture(scope="function")
def test_app():
    app = Flask("Scan-o-Matic UI", template_folder=Paths().ui_templates)
    rpc_client = Mock()
    experiment_api.add_routes(app, rpc_client, logging)
    app.testing = True

    test_app = app.test_client()
    test_app.rpc_client = rpc_client

    return test_app


class TestFeatureExtractEndpoint:

    route = '/feature_extract'

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

        response = test_app.post(
            self.route + '?action=extract',
            data={
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
        assert json.loads(response.data)['success']

    @patch(
        'scanomatic.ui_server.experiment_api.FeaturesFactory._validate_analysis_directory',
        return_value=True)
    def test_dont_keep_previous_qc(self, validator_mock, test_app):

        test_app.rpc_client.create_feature_extract_job.return_value = 'Hi'

        response = test_app.post(
            self.route + '?action=extract',
            data={
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
        assert json.loads(response.data)['success']
