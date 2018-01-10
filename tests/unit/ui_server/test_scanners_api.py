import pytest
from flask import Flask

from scanomatic.ui_server.ui_server import add_configs
from scanomatic.io.paths import Paths
from scanomatic.ui_server import scanners_api


@pytest.fixture
def app():
    app = Flask(__name__, template_folder=Paths().ui_templates)
    scanners_api.add_routes(app)
    return app


@pytest.fixture(scope="function")
def test_app(app):
    add_configs(app)
    test_app = app.test_client()
    return test_app


class TestScannerStatus:

    URI = '/api/scanners'

    def test_get_all_implicit(self, test_app):
        response = test_app.get(self.URI)
        response.status_code == 200
        assert response.json == {
            'scanners': [{'name': 'Test', 'owner': None, 'power': True}]
        }

    def test_get_free_scanners(self, test_app):
        response = test_app.get(self.URI + '?free=1')
        response.status_code == 200
        assert response.json == {
            'scanners': [{'name': 'Test', 'owner': None, 'power': True}]
        }

    def test_get_scanner(self, test_app):
        response = test_app.get(self.URI + "/Test")
        response.status_code == 200
        assert response.json == {
            'scanner': {'name': 'Test', 'owner': None, 'power': True}
        }

    def test_get_unknown_scanner(self, test_app):
        response = test_app.get(self.URI + "/Unknown")
        response.status_code == 400
        assert response.json['reason'] == "Scanner 'Unknown' unknown"
