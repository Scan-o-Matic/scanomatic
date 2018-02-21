from __future__ import absolute_import

import uuid
import requests


def test_scanner_is_listed_after_added(scanomatic, browser):
    scannerid = create_scanner(scanomatic)
    response = requests.get(scanomatic + '/api/scanners')
    scanners = response.json()
    scanner = [
        scanner for scanner in scanners if scanner['identifier'] == scannerid
    ]
    assert len(scanner) == 1
    scanner = scanner[0]
    name = scanner['name']
    assert len(name.split(' ', 2)) == 3
    assert name.split(' ', 2)[0] == 'The'


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
