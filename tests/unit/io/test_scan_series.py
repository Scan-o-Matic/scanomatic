from __future__ import absolute_import

import pytest
from scanomatic.io.scan_series import (
    ScanSeries, ScanNameCollision, ScanNameUnknown
)


@pytest.fixture(scope='function')
def scan_series():
    return ScanSeries()


class TestAddJob:

    def test_add_jobb(self, scan_series):
        name = "Test"
        job = {'hello': 'to you'}
        scan_series.add_job(name, job)
        assert job in scan_series.get_jobs()

    def test_add_duplicate_job_raises(self, scan_series):
        name = "Test"
        job = {'hello': 'to you'}
        scan_series.add_job(name, job)
        with pytest.raises(ScanNameCollision):
            scan_series.add_job(name, job)


class TestRemoveJob:

    def test_remove_job(self, scan_series):
        name = "Test"
        job = {'hello': 'to you'}
        scan_series.add_job(name, job)
        scan_series.remove_job(name)
        assert job not in scan_series.get_jobs()

    def test_remove_unknown_job_raises(self, scan_series):
        with pytest.raises(ScanNameUnknown):
            scan_series.remove_job("Help")


class TestGetJobs:

    def test_when_no_jobs(self, scan_series):
        assert scan_series.get_jobs() == []

    def test_get_all_jobs(self, scan_series):
        job1 = [42]
        job2 = {6: 7}
        scan_series.add_job('a', job1)
        scan_series.add_job('b', job2)
        assert job1 in scan_series.get_jobs()
        assert job2 in scan_series.get_jobs()
        assert len(scan_series.get_jobs()) == 2
