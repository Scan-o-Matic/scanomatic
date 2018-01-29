from __future__ import absolute_import
from httplib import OK, NOT_FOUND, BAD_REQUEST, CREATED
import json

from flask import Flask
import pytest
import freezegun

from scanomatic.io.paths import Paths
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
    SCANNER_TWO = {
        u'name': u'Scanner one',
        u'identifier': u'9a8486a6f9cb11e7ac660050b68338ac',
    }

    SCANNER_ONE = {
        u'name': u'Scanner two',
        u'identifier': u'350986224086888954',
    }

    def test_get_all_implicit(self, test_app):
        response = test_app.get(self.URI)
        assert response.status_code == OK
        assert len(response.json) == 2
        assert all(
            scanner in response.json
            for scanner in [self.SCANNER_ONE, self.SCANNER_TWO]
        )

    def test_get_free_scanners(self, test_app):
        response = test_app.get(self.URI + '?free=1')
        assert response.status_code == OK
        assert len(response.json) == 2
        assert all(
            scanner in response.json
            for scanner in [self.SCANNER_ONE, self.SCANNER_TWO]
        )

    def test_get_scanner(self, test_app):
        response = test_app.get(self.URI + "/9a8486a6f9cb11e7ac660050b68338ac")
        assert response.status_code == OK
        assert response.json == self.SCANNER_TWO

    def test_get_unknown_scanner(self, test_app):
        response = test_app.get(self.URI + "/Unknown")
        assert response.status_code == NOT_FOUND
        assert response.json['reason'] == "Scanner 'Unknown' unknown"

    def test_add_scanner_status(self, test_app):
        with freezegun.freeze_time('1985-10-26 01:20', tz_offset=0):
            response = test_app.put(
                self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
                data=json.dumps({u"job": u"foo"}),
                headers={'Content-Type': 'application/json'}
            )
            assert response.status_code == OK

            response = test_app.get(
                self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status")
            assert response.status_code == OK
            assert response.json["job"] == "foo"
            assert response.json["serverTime"] == "1985-10-26T01:20:00Z"

    def test_get_scanner_status(self, test_app):
        response = test_app.get(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status")
        assert response.status_code == OK
        assert response.json == {u'job': None, u'power': False, u'owner': None}

    def test_get_unknown_scanner_status_fails(self, test_app):
        response = test_app.get(self.URI + "/42/status")
        assert response.status_code == NOT_FOUND

    def test_add_bad_scanner_status_fails(self, test_app):
        response = test_app.put(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
            data=json.dumps({"foo": "foo", "bar": "bar"}),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == BAD_REQUEST

    def test_add_unknown_scanner_status(self, test_app):
        response = test_app.put(
            self.URI + "/42/status",
            data=json.dumps({"job": "foo"}),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == CREATED
