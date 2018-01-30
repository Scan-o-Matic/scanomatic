from __future__ import absolute_import

from httplib import OK, NOT_FOUND

from freezegun import freeze_time
from flask import Flask


class TestGetScannerJob(object):
    SCANNERID = '9a8486a6f9cb11e7ac660050b68338ac'

    def test_invalid_scanner(self, apiclient):
        response = apiclient.get_scanner_job('xxxx')
        assert response.status_code == NOT_FOUND

    def test_has_no_scanjob(self, apiclient):
        response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == OK
        assert response.json is None

    def test_has_scanjob(self, apiclient):
        jobid = apiclient.create_scan_job(self.SCANNERID).json['identifier']
        job = apiclient.get_scan_job(jobid).json
        with freeze_time('1985-10-26 01:20', tz_offset=0):
            apiclient.start_scan_job(jobid)
        with freeze_time('1985-10-26 01:21', tz_offset=0):
            response = apiclient.get_scanner_job(self.SCANNERID)
        assert response.status_code == OK
        assert response.json == dict(startTime='1985-10-26T01:20:00Z', **job)
