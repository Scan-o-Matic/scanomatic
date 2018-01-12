from __future__ import absolute_import
import pytest
from flask import Flask
from httplib import OK, NOT_FOUND

from scanomatic.ui_server.ui_server import add_configs
from scanomatic.io.paths import Paths
from scanomatic.ui_server import scanners_api


@pytest.fixture
def app():
    app = Flask(__name__, template_folder=Paths().ui_templates)
    app.register_blueprint(
        scanners_api.blueprint, url_prefix="/api/scanners"
    )
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
        response.status_code == OK
        assert response.json == [{
            'name': 'Test', 'owner': None, 'power': False
        }]

    def test_get_free_scanners(self, test_app):
        response = test_app.get(self.URI + '?free=1')
        response.status_code == OK
        assert response.json == [{
            'name': 'Test', 'owner': None, 'power': False
        }]

    def test_get_scanner(self, test_app):
        response = test_app.get(self.URI + "/Test")
        response.status_code == OK
        assert response.json == {
            'name': 'Test', 'owner': None, 'power': False
        }

    def test_get_unknown_scanner(self, test_app):
        response = test_app.get(self.URI + "/Unknown")
        response.status_code == NOT_FOUND
        assert response.json['reason'] == "Scanner 'Unknown' unknown"
