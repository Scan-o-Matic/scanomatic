from __future__ import absolute_import
import json
import uuid

import pytest
from flask import Flask

from scanomatic.io.imagestore import ImageStore
from scanomatic.io.scanning_store import ScanningStore
from scanomatic.io.scannerstore import ScannerStore
from scanomatic.models.scanner import Scanner
from scanomatic.ui_server import scan_jobs_api
from scanomatic.ui_server import scanners_api
from scanomatic.ui_server import scans_api
import scanomatic.ui_server.database as db


@pytest.fixture
def app(tmpdir, database):
    app = Flask(__name__)
    app.register_blueprint(scanners_api.blueprint, url_prefix="/scanners")
    app.register_blueprint(scan_jobs_api.blueprint, url_prefix="/scan-jobs")
    app.register_blueprint(scans_api.blueprint, url_prefix="/scans")
    app.config['DATABASE_URL'] = database
    db.setup(app)
    app.config['scanning_store'] = ScanningStore()
    app.config['imagestore'] = ImageStore(str(tmpdir))
    with app.app_context():
        scannerstore = ScannerStore(db.connect())
        scannerstore.add(
            Scanner('Scanner one', '9a8486a6f9cb11e7ac660050b68338ac')
        )
        scannerstore.add(
            Scanner('Scanner two', '350986224086888954')
        )
    return app


@pytest.fixture
def apiclient(client):
    class APIClient:
        def create_scan_job(
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

        def get_scan_job(self, jobid):
            return client.get('/scan-jobs/{}'.format(jobid))

        def start_scan_job(self, jobid):
            return client.post('/scan-jobs/{}/start'.format(jobid))

        def get_scanner_job(self, scannerid):
            return client.get('/scanners/{}/job'.format(scannerid))

        def post_scan(self, data):
            return client.post('/scans', data=data)

        def get_scans(self):
            return client.get('/scans')

        def get_scan(self, scanid):
            return client.get('/scans/{}'.format(scanid))

        def get_scan_image(self, scanid):
            return client.get('/scans/{}/image'.format(scanid))

    return APIClient()
