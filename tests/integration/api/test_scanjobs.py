from __future__ import absolute_import
from httplib import BAD_REQUEST, CONFLICT, OK, CREATED, NOT_FOUND
import json
from types import MethodType
import uuid

import freezegun
import pytest


@pytest.fixture(scope="function")
def test_app(client):
    def _post_json(self, uri, data, **kwargs):
        return self.post(
            uri,
            data=json.dumps(data),
            content_type='application/json',
            **kwargs
        )
    client.post_json = MethodType(_post_json, client)
    return client


class TestScanJobs:

    URI = '/scan-jobs'

    @pytest.fixture(scope='function')
    def job(self):
        return {
            'name': 'Binary yeast',
            'scannerId': '9a8486a6f9cb11e7ac660050b68338ac',
            'interval': 500,
            'duration': 86400,
        }

    def test_get_jobs_and_there_are_none(self, test_app):
        response = test_app.get(self.URI)
        response.status_code == OK
        assert response.json == []

    def test_get_unknown_job(self, test_app):
        response = test_app.get(self.URI + '/unknown')
        assert response.status_code == NOT_FOUND

    def test_add_job(self, test_app, job):
        response = test_app.post_json(self.URI, job)
        assert response.status_code == CREATED
        jobid = response.json['identifier']
        response2 = test_app.get(self.URI + '/' + jobid)
        assert response2.status_code == OK
        assert response2.json == {
            'identifier': jobid,
            'name': job['name'],
            'interval': job['interval'],
            'duration': job['duration'],
            'scannerId': job['scannerId'],
        }

    def test_sereval_identical_job_names_fails(self, test_app, job):
        response = test_app.post_json(self.URI, job)
        assert response.status_code == CREATED
        response = test_app.post_json(self.URI, job)
        assert response.status_code == CONFLICT
        assert response.json['reason'] == "Name 'Binary yeast' duplicated"

    @pytest.mark.parametrize("key,reason", (
        ('name', 'No name supplied'),
        ('duration', 'Duration not supplied'),
        ('interval', 'Interval not supplied'),
        ('scannerId', 'Scanner not supplied'),
    ))
    def test_add_job_without_info(self, test_app, job, key, reason):
        del job[key]
        response = test_app.post_json(self.URI, job)
        assert response.status_code == BAD_REQUEST
        assert response.json['reason'] == reason

    def test_add_with_too_short_interval(self, test_app, job):
        job['interval'] = 1
        response = test_app.post_json(self.URI, job)
        assert response.status_code == BAD_REQUEST
        assert response.json['reason'] == 'Interval too short'

    def test_add_with_unknown_scanner(self, test_app, job):
        job['scannerId'] = "unknown"
        response = test_app.post_json(self.URI, job)
        assert response.status_code == BAD_REQUEST
        assert response.json['reason'] == "Scanner 'unknown' unknown"

    def test_added_job_gets_listed(self, test_app, job):
        response = test_app.post_json(self.URI, job)
        assert response.status_code == CREATED
        identifier = response.json['identifier']
        response = test_app.get(self.URI)
        response.status_code == OK
        assert response.json == [
            {
                'identifier': identifier,
                'name': job['name'],
                'interval': job['interval'],
                'duration': job['duration'],
                'scannerId': job['scannerId'],
            }
        ]

    def test_cant_store_bogus_setttings(self, test_app, job):
        job['bogus'] = True
        response = test_app.post_json(self.URI, job)
        assert response.status_code == CREATED
        identifier = response.json['identifier']
        response = test_app.get(self.URI)
        response.status_code == OK
        assert response.json == [
            {
                'identifier': identifier,
                'name': job['name'],
                'interval': job['interval'],
                'duration': job['duration'],
                'scannerId': job['scannerId'],
            }
        ]


class TestStartScanJob:
    URI = '/scan-jobs/{id}/start'

    def create_scanjob(self, test_app):
        response = test_app.post_json('/scan-jobs', {
            'name': uuid.uuid1().hex,
            'scannerId': '9a8486a6f9cb11e7ac660050b68338ac',
            'interval': 500,
            'duration': 86400,
        })
        assert response.status_code == CREATED
        return response.json['identifier']

    def test_start_existing_job(self, test_app):
        jobid = self.create_scanjob(test_app)
        with freezegun.freeze_time('1985-10-26 01:20', tz_offset=0):
            response = test_app.post(self.URI.format(id=jobid))
        assert response.status_code == OK
        job = test_app.get('/scan-jobs/' + jobid).json
        return job['startTime'] == '1985-10-26T01:20:00Z'

    def test_already_started(self, test_app):
        jobid = self.create_scanjob(test_app)
        with freezegun.freeze_time('1985-10-26 01:20', tz_offset=0):
            test_app.post(self.URI.format(id=jobid))
        with freezegun.freeze_time('1985-10-26 01:21', tz_offset=0):
            response = test_app.post(self.URI.format(id=jobid))
        assert response.status_code == CONFLICT
        job = test_app.get('/scan-jobs/' + jobid).json
        assert job['startTime'] == '1985-10-26T01:20:00Z'

    def test_unknown_job(self, test_app):
        response = test_app.post(self.URI.format(id='unknown'))
        assert response.status_code == NOT_FOUND

    def test_scanner_busy(self, test_app):
        jobid1 = self.create_scanjob(test_app)
        test_app.post(self.URI.format(id=jobid1))
        jobid2 = self.create_scanjob(test_app)
        response = test_app.post(self.URI.format(id=jobid2))
        assert response.status_code == CONFLICT
