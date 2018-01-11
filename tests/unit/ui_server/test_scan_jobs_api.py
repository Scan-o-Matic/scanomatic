from __future__ import absolute_import
import pytest
import json
from types import MethodType
from flask import Flask

from scanomatic.ui_server.ui_server import add_configs
from scanomatic.io.paths import Paths
from scanomatic.ui_server import scan_jobs_api


@pytest.fixture
def app():
    app = Flask(__name__, template_folder=Paths().ui_templates)
    scan_jobs_api.add_routes(app)
    return app


@pytest.fixture(scope="function")
def test_app(app):
    def _post_json(self, uri, data, **kwargs):
        return self.post(
            uri,
            data=json.dumps(data),
            content_type='application/json',
            **kwargs
        )
    add_configs(app)
    test_app = app.test_client()
    test_app.post_json = MethodType(_post_json, test_app)
    return test_app


class TestScanJobs:

    URI = '/api/scan-jobs'

    @pytest.fixture(scope='function')
    def job(self):
        return {
            'name': 'Binary yeast',
            'scannerName': 'Test',
            'interval': 32,
            'duration': {
                'days': 1024,
                'hours': 64,
                'minutes': 8,
            }
        }

    def test_get_jobs_and_there_are_none(self, test_app):
        response = test_app.get(self.URI)
        response.status_code == 200
        assert response.json == {'jobs': []}

    def test_add_job(self, test_app, job):
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 200

    def test_sereval_identical_job_names_fails(self, test_app, job):
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 200
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 400
        assert response.json['reason'] == "Name 'Binary yeast' duplicated"

    @pytest.mark.parametrize("key,reason", (
        ('name', 'No name supplied'),
        ('duration', 'Duration not supplied'),
        ('interval', 'Interval not supplied'),
        ('scannerName', 'Scanner not supplied'),
    ))
    def test_add_job_without_info(self, test_app, job, key, reason):
        del job[key]
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 400
        assert response.json['reason'] == reason

    def test_add_with_too_short_interval(self, test_app, job):
        job['interval'] = 1
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 400
        assert response.json['reason'] == 'Interval too short'

    def test_add_with_unknown_scanner(self, test_app, job):
        job['scannerName'] = "unknown"
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 400
        assert response.json['reason'] == "Scanner 'unknown' unknown"

    def test_added_job_gets_listed(self, test_app, job):
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 200
        identifier = response.json['jobId']
        response = test_app.get(self.URI)
        response.status_code == 200
        assert response.json == {'jobs': [
            {
                'id': identifier,
                'name': job['name'],
                'interval': job['interval'],
                'duration': job['duration'],
                'scanner': {
                    'name': job['scannerName'],
                    'power': False,
                    'owner': None,
                }
            }
        ]}

    def test_cant_store_bogus_setttings(self, test_app, job):
        job['bogus'] = True
        response = test_app.post_json(self.URI, job)
        assert response.status_code == 200
        identifier = response.json['jobId']
        response = test_app.get(self.URI)
        response.status_code == 200
        assert response.json == {'jobs': [
            {
                'id': identifier,
                'name': job['name'],
                'interval': job['interval'],
                'duration': job['duration'],
                'scanner': {
                    'name': job['scannerName'],
                    'power': False,
                    'owner': None,
                }
            }
        ]}
