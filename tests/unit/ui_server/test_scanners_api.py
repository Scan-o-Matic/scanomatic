from __future__ import absolute_import
from datetime import datetime, timedelta
from httplib import OK, NOT_FOUND

from flask import Flask
import mock
import pytest

from scanomatic.io.paths import Paths
from scanomatic.models.scanjob import ScanJob
from scanomatic.ui_server import scanners_api
from scanomatic.ui_server.ui_server import add_configs


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
    SCANNER = {
        'name': 'Test',
        'owner': None,
        'power': False,
        'identifier': '9a8486a6f9cb11e7ac660050b68338ac',
    }

    def test_get_all_implicit(self, test_app):
        response = test_app.get(self.URI)
        response.status_code == OK
        assert response.json == [self.SCANNER]

    def test_get_free_scanners(self, test_app):
        response = test_app.get(self.URI + '?free=1')
        response.status_code == OK
        assert response.json == [self.SCANNER]

    def test_get_scanner(self, test_app):
        response = test_app.get(self.URI + "/9a8486a6f9cb11e7ac660050b68338ac")
        response.status_code == OK
        assert response.json == self.SCANNER

    def test_get_unknown_scanner(self, test_app):
        response = test_app.get(self.URI + "/Unknown")
        response.status_code == NOT_FOUND
        assert response.json['reason'] == "Scanner 'Unknown' unknown"


class TestGetScannerJob(object):
    URI = '/api/scanners/xxxx/job'

    @pytest.fixture
    def fakedb(self):
        return mock.MagicMock()

    @pytest.fixture
    def app(self, fakedb):
        app = Flask(__name__, template_folder=Paths().ui_templates)
        app.config['scanning_store'] = fakedb
        app.register_blueprint(
            scanners_api.blueprint, url_prefix="/api/scanners"
        )
        return app

    def test_invalid_scanner(self, fakedb, client):
        fakedb.has_scanner.return_value = False
        response = client.get(self.URI)
        assert response.status_code == NOT_FOUND

    def test_has_no_scanjob(self, fakedb, client):
        fakedb.has_scanner.return_value = True
        fakedb.get_current_scanjob.return_value = None
        response = client.get(self.URI)
        assert response.status_code == OK
        assert response.json is None

    def test_has_scanjob(self, fakedb, client):
        fakedb.has_scanner.return_value = True
        fakedb.get_current_scanjob.return_value = ScanJob(
            identifier='xxxx',
            name='The Job',
            duration=timedelta(days=3),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
            start=datetime(1985, 10, 26, 1, 20),
        )
        response = client.get(self.URI)
        assert response.status_code == OK
        assert response.json == {
            'id': 'xxxx',
            'name': 'The Job',
            'duration': 259200,
            'interval': 300,
            'scannerId': 'yyyy',
            'start': '1985-10-26T01:20:00',
        }
        pass
