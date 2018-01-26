from __future__ import absolute_import
from datetime import datetime, timedelta
from httplib import OK, NOT_FOUND, BAD_REQUEST
import json

from flask import Flask
import pytest
import freezegun
from pytz import utc

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
    SCANNER_OFF = {
        u'name': u'Never On',
        u'power': False,
        u'identifier': u'9a8486a6f9cb11e7ac660050b68338ac',
    }

    SCANNER_ON = {
        u'name': u'Always On',
        u'power': True,
        u'identifier': u'350986224086888954',
    }

    def test_get_all_implicit(self, test_app):
        response = test_app.get(self.URI)
        assert response.status_code == OK
        assert len(response.json) == 2
        assert all(
            scanner in response.json
            for scanner in [self.SCANNER_ON, self.SCANNER_OFF]
        )

    def test_get_free_scanners(self, test_app):
        response = test_app.get(self.URI + '?free=1')
        assert response.status_code == OK
        assert len(response.json) == 2
        assert all(
            scanner in response.json
            for scanner in [self.SCANNER_ON, self.SCANNER_OFF]
        )

    def test_get_scanner(self, test_app):
        response = test_app.get(self.URI + "/9a8486a6f9cb11e7ac660050b68338ac")
        assert response.status_code == OK
        assert response.json == self.SCANNER_OFF

    def test_get_unknown_scanner(self, test_app):
        response = test_app.get(self.URI + "/Unknown")
        assert response.status_code == NOT_FOUND
        assert response.json['reason'] == "Scanner 'Unknown' unknown"
