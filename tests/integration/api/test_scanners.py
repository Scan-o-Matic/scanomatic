from __future__ import absolute_import

import httplib as HTTPStatus
import json

from freezegun import freeze_time
import mock
import pytest


class TestGetScannerJob(object):
    SCANNERID = '9a8486a6f9cb11e7ac660050b68338ac'

    def test_invalid_scanner(self, apiclient):
        response = apiclient.get_scanner_job('xxxx')
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_has_no_scanjob(self, apiclient):
        response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == HTTPStatus.OK
        assert response.json is None

    def test_terminated_job(self, apiclient):
        jobid = apiclient.create_scan_job(
            self.SCANNERID, duration=600
        ).json['identifier']
        with freeze_time('1985-10-26 01:20:00Z'):
            apiclient.start_scan_job(jobid)
        with freeze_time('1985-10-26 01:21:00Z'):
            apiclient.terminate_scan_job(jobid)
        with freeze_time('1985-10-26 01:22:00Z'):
            response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == HTTPStatus.OK
        assert response.json is None

    def test_has_scanjob(self, apiclient):
        jobid = apiclient.create_scan_job(self.SCANNERID).json['identifier']
        job = apiclient.get_scan_job(jobid).json
        with freeze_time('1985-10-26 01:20', tz_offset=0):
            apiclient.start_scan_job(jobid)
        with freeze_time('1985-10-26 01:21', tz_offset=0):
            response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == HTTPStatus.OK
        assert response.json == dict(startTime='1985-10-26T01:20:00Z', **job)


class TestScannerStatus:

    URI = '/scanners'
    SCANNER_ONE = {
        u'name': u'Scanner one',
        u'identifier': u'9a8486a6f9cb11e7ac660050b68338ac',
        u'power': False,
    }

    SCANNER_TWO = {
        u'name': u'Scanner two',
        u'identifier': u'350986224086888954',
        u'power': False,
    }

    def test_get_all_implicit(self, client):
        response = client.get(self.URI)
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 2
        assert all(
            scanner in response.json
            for scanner in [self.SCANNER_TWO, self.SCANNER_ONE]
        )

    def test_get_scanner(self, client):
        response = client.get(self.URI + "/9a8486a6f9cb11e7ac660050b68338ac")
        assert response.status_code == HTTPStatus.OK
        assert response.json == self.SCANNER_ONE

    def test_get_unknown_scanner(self, client):
        response = client.get(self.URI + "/Unknown")
        assert response.status_code == HTTPStatus.NOT_FOUND
        assert response.json['reason'] == "Scanner 'Unknown' unknown"

    @pytest.fixture
    def jsonstatus(self):
        return {
            'job': 'curr3ntj0b',
            'imagesToSend': 2,
            'startTime': '1985-10-26T00:00:00Z',
            'nextScheduledScan': '1985-10-26T00:22:00Z',
            'devices': ['epson'],
        }

    def test_add_scanner_no_scanners_status(self, client, jsonstatus):
        jsonstatus['devices'] = []
        response = client.put(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
            data=json.dumps(jsonstatus),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == HTTPStatus.OK

    def test_add_scanner_status(self, client, jsonstatus):
        with freeze_time('1985-10-26 01:20', tz_offset=0):
            response = client.put(
                self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
                data=json.dumps(jsonstatus),
                headers={'Content-Type': 'application/json'}
            )
            assert response.status_code == HTTPStatus.OK

            response = client.get(
                self.URI + "/9a8486a6f9cb11e7ac660050b68338ac")
            assert response.status_code == HTTPStatus.OK
            assert response.json["power"] is True

    @pytest.mark.parametrize('status', [
        {
            'imagesToSend': 0,
            'startTime': '1985-10-26T00:00:00Z',
            'devices': ['epson'],
        },
        {
            'imagesToSend': 0,
            'startTime': '1985-10-26T00:00:00Z',
            'job': None,
            'nextScheduledScan': None,
            'devices': ['epson'],
        },
    ])
    def test_add_scanner_status_no_job(self, client, status):
        response = client.put(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
            data=json.dumps(status),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == HTTPStatus.OK

    @pytest.mark.parametrize('property, value', [
        ('imagesToSend', 'x'),
        ('imagesToSend', -5),
        ('startTime', 'xxx'),
    ])
    def test_put_invalid_value(self, client, jsonstatus, property, value):
        jsonstatus[property] = value
        response = client.put(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
            data=json.dumps(jsonstatus),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST

    @pytest.mark.parametrize('property', ['imagesToSend', 'startTime'])
    def test_put_missing_property(self, client, jsonstatus, property):
        del jsonstatus[property]
        response = client.put(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
            data=json.dumps(jsonstatus),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_add_bad_scanner_status_fails(self, client):
        response = client.put(
            self.URI + "/9a8486a6f9cb11e7ac660050b68338ac/status",
            data=json.dumps({"foo": "foo", "bar": "bar"}),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_add_unknown_scanner_status(self, client, jsonstatus):
        response = client.put(
            self.URI + "/42/status",
            data=json.dumps(jsonstatus),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == HTTPStatus.CREATED

    def test_add_unkown_scanner_status_duplicate_name(
        self, client, jsonstatus,
    ):
        with mock.patch(
            'scanomatic.scanning.update_scanner_status.get_generic_name',
            return_value="Scanner two"
        ):
            response = client.put(
                self.URI + "/42/status",
                data=json.dumps(jsonstatus),
                headers={'Content-Type': 'application/json'}
            )
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
