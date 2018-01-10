import pytest
from mock import MagicMock
from flask import Flask

from scanomatic.ui_server.ui_server import add_configs
from scanomatic.io.paths import Paths
from scanomatic.ui_server import status_api


@pytest.fixture
def rpc_client():
    return MagicMock()


@pytest.fixture
def app(rpc_client):
    app = Flask(__name__, template_folder=Paths().ui_templates)
    status_api.add_routes(app, rpc_client)
    return app


@pytest.fixture(scope="function")
def test_app(app, rpc_client):
    add_configs(app)
    test_app = app.test_client()
    test_app.rpc_client = rpc_client
    return test_app


class TestScannerStatus:

    @pytest.mark.parametrize("uri", (
        '/api/status/scanners',
        '/api/status/scanners/all',
    ))
    def test_get_all_implicit(self, test_app, uri):
        response = test_app.get(uri)
        response.status_code == 200
        assert response.json == {
            'scanners': [{'name': 'Test', 'owner': None, 'power': True}]
        }

    def test_get_free_scanners(self, test_app):
        uri = '/api/status/scanners/free'
        response = test_app.get(uri)
        response.status_code == 200
        assert response.json == {
            'scanners': [{'name': 'Test', 'owner': None, 'power': True}]
        }

    def test_get_scanner(self, test_app):
        uri = '/api/status/scanners/Test'
        response = test_app.get(uri)
        response.status_code == 200
        assert response.json == {
            'scanner': {'name': 'Test', 'owner': None, 'power': True}
        }

    def test_get_unknown_scanner(self, test_app):
        uri = '/api/status/scanners/Unknown'
        response = test_app.get(uri)
        response.status_code == 400
        assert response.json['reason'] == "Scanner 'Unknown' unknown"
