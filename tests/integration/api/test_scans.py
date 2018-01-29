from __future__ import absolute_import

from hashlib import sha256
import httplib as HTTPStatus
from io import BytesIO

import pytest


@pytest.fixture
def existing_scanjob_id(apiclient):
    scannerid = '9a8486a6f9cb11e7ac660050b68338ac'
    response = apiclient.create_scan_job(scannerid)
    assert response.status_code == HTTPStatus.CREATED
    return response.json['identifier']


@pytest.fixture
def scan_data(existing_scanjob_id):
    image = b'foobar'
    return {
        'image': (BytesIO(image), 'image.tiff'),
        'digest': 'sha256:' + sha256(image).hexdigest(),
        'startTime': '1985-10-26T01:20:00Z',
        'endTime': '1985-10-26T01:21:00Z',
        'scanJobId': existing_scanjob_id
    }


class TestPostScan(object):
    def test_accept_scan_data(self, apiclient, scan_data):
        response = apiclient.post_scan(scan_data)
        assert response.status_code == HTTPStatus.CREATED, response.data

    def test_accept_bad_digest(self, apiclient, scan_data):
        scan_data['digest'] = 'sha256:xxx'
        response = apiclient.post_scan(scan_data)
        assert response.status_code == HTTPStatus.CREATED, response.data

    @pytest.mark.parametrize('key', [
        'image', 'digest', 'startTime', 'endTime', 'scanJobId',
    ])
    def test_reject_incomplete_data(self, apiclient, scan_data, key):
        del scan_data[key]
        response = apiclient.post_scan(scan_data)
        assert response.status_code == HTTPStatus.BAD_REQUEST

    @pytest.mark.parametrize('key, value', [
        ('image', 'foobar'),
        ('digest', ''),
        ('digest', 'xxxx'),
        ('startTime', 'xxxx'),
        ('endTime', 'xxxx'),
        ('scanJobId', 'unknown'),
    ])
    def test_reject_invalid_data(self, apiclient, scan_data, key, value):
        scan_data[key] = value
        response = apiclient.post_scan(scan_data)
        assert response.status_code == HTTPStatus.BAD_REQUEST


class TestGetScans(object):
    def test_get_scans_empty_list(self, apiclient):
        response = apiclient.get_scans()
        assert response.json == []

    def test_existing_scans(self, apiclient, scan_data):
        apiclient.post_scan(scan_data)
        response2 = apiclient.get_scans()
        assert response2.status_code == HTTPStatus.OK
        assert len(response2.json) == 1
        assert response2.json[0]['id']
        assert response2.json[0]['startTime'] == scan_data['startTime']
        assert response2.json[0]['endTime'] == scan_data['endTime']
        assert response2.json[0]['digest'] == scan_data['digest']
        assert response2.json[0]['scanJobId'] == scan_data['scanJobId']


class TestGetScan(object):
    def test_unknown_scan_id(self, apiclient):
        response = apiclient.get_scan('unknown')
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_existing_scan(self, apiclient, scan_data):
        post_response = apiclient.post_scan(scan_data)
        print(post_response.json)
        scanid = post_response.json['identifier']
        response = apiclient.get_scan(scanid)
        assert response.status_code == HTTPStatus.OK
        assert response.json['id']
        assert response.json['startTime'] == scan_data['startTime']
        assert response.json['endTime'] == scan_data['endTime']
        assert response.json['digest'] == scan_data['digest']
        assert response.json['scanJobId'] == scan_data['scanJobId']


class TestGetScanImage(object):
    def test_unknown_scan_id(self, apiclient):
        response = apiclient.get_scan_image('unknown')
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_existing_scan(self, apiclient, scan_data):
        post_response = apiclient.post_scan(scan_data)
        scanid = post_response.json['identifier']
        response = apiclient.get_scan_image(scanid)
        assert response.status_code == HTTPStatus.OK
        assert response.data == b'foobar'
        print(response.headers)
        assert response.headers['Content-Type'] == 'image/tiff'
