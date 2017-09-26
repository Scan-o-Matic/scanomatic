import pytest
import os

from flask import Flask
import mock

from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config
from scanomatic.ui_server import ui_server


@pytest.fixture(scope="function")
def test_app():
    app = Flask("Scan-o-Matic UI", template_folder=Paths().ui_templates)
    ui_server.add_routes(app)
    app.testing = True
    return app.test_client()


class TestFeatureExtractEndpoint:

    route = '/feature_extract'

    @staticmethod
    def jailed_path(path):
        return os.path.abspath(
            path.replace('root', Config().paths.projects_root, 1))

    @mock.patch('ui_server.get_client')
    def test_keep_previous_qc(self, test_app, my_mock):

        my_mock.return_value.create_feature_extract_job.return_value = 'Hi'

        response = test_app.post(
            self.route,
            data={
                'action': 'extract',
                'analysis_directory': 'root/test/',
                'keep_qc': 1,
            },
            follow_redirects=True
        )
        assert my_mock.return_value.create_feature_extract_job.called
        assert my_mock.return_value.create_feature_extract_job.called_with({
            'try_keep_qc': True,
            'analysis_directory': self.jailed_path('root/test/')
        })
        assert response.status_code == 200
        assert response.data['success']
