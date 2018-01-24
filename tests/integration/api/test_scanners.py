from __future__ import absolute_import

from httplib import BAD_REQUEST, CONFLICT, INTERNAL_SERVER_ERROR, CREATED, OK, NOT_FOUND
import json
import uuid

from freezegun import freeze_time
import pytest
from flask import Flask

from scanomatic.ui_server import scanners_api
from scanomatic.ui_server import scan_jobs_api
from scanomatic.ui_server.ui_server import add_configs


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(scanners_api.blueprint, url_prefix="/scanners")
    app.register_blueprint(scan_jobs_api.blueprint, url_prefix="/scan-jobs")
    add_configs(app)
    return app


@pytest.fixture
def apiclient(client):
    class APIClient:
        def create_scanning_job(
            self, scannerid, name=None, duration=600, interval=300,
        ):
            if name is None:
                name = uuid.uuid1().hex
            return client.post(
                '/scan-jobs',
                data=json.dumps({
                    'name': name,
                    'duration': duration,
                    'interval': interval,
                    'scannerId': scannerid,
                }),
                content_type='application/json',
            )

        def start_scanning_job(self, jobid):
            return client.post('/scan-jobs/{}/start'.format(jobid))

        def get_scanner_job(self, scannerid):
            return client.get('/scanners/{}/job'.format(scannerid))

    return APIClient()


class TestGetScannerJob(object):
    URI = '/api/scanners/{}/job'
    SCANNERID = '9a8486a6f9cb11e7ac660050b68338ac'

    def test_invalid_scanner(self, apiclient):
        response = apiclient.get_scanner_job('xxxx')
        assert response.status_code == NOT_FOUND

    def test_has_no_scanjob(self, apiclient):
        response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == OK
        assert response.json is None

    def test_has_scanjob(self, apiclient):
        jobid = apiclient.create_scanning_job(self.SCANNERID).json['jobId']
        with freeze_time('1985-10-26 01:20', tz_offset=0):
            apiclient.start_scanning_job(jobid)
        with freeze_time('1985-10-26 01:21', tz_offset=0):
            response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == OK
        assert response.json['identifier'] == jobid
        assert response.json['scannerId'] == self.SCANNERID
        assert response.json['startTime'] == '1985-10-26T01:20:00Z'
