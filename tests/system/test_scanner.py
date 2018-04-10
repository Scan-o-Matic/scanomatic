from __future__ import absolute_import

import uuid
import requests


def test_added_scanner_gets_a_name(scanomatic):
    scannerid = create_scanner(scanomatic)
    response = requests.get(scanomatic + '/api/scanners/' + scannerid)
    response.raise_for_status()
    scanner = response.json()
    name = scanner['name']
    assert len(name.split(' ', 1)) == 2


def create_scanner(scanomatic):
    scannerid = uuid.uuid4().hex
    response = requests.put(
        scanomatic + '/api/scanners/{}/status'.format(scannerid),
        json={
            'startTime': '2000-01-02T00:00:00Z',
            'imagesToSend': 0,
            'devices': ['epson'],
        },
    )
    response.raise_for_status()
    return scannerid
