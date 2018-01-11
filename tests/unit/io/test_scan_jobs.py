from __future__ import absolute_import

import pytest
from scanomatic.io.scan_jobs import (
    ScanJobs, ScanNameCollision, ScanNameUnknown
)


@pytest.fixture(scope='function')
def scan_jobs():
    return ScanJobs()


class TestAddJob:

    def test_add_jobb(self, scan_jobs):
        name = "Test"
        job = {'hello': 'to you'}
        scan_jobs.add_job(name, job)
        assert job in scan_jobs.get_jobs()

    def test_add_duplicate_job_raises(self, scan_jobs):
        name = "Test"
        job = {'hello': 'to you'}
        scan_jobs.add_job(name, job)
        with pytest.raises(ScanNameCollision):
            scan_jobs.add_job(name, job)


class TestRemoveJob:

    def test_remove_job(self, scan_jobs):
        name = "Test"
        job = {'hello': 'to you'}
        scan_jobs.add_job(name, job)
        scan_jobs.remove_job(name)
        assert job not in scan_jobs.get_jobs()

    def test_remove_unknown_job_raises(self, scan_jobs):
        with pytest.raises(ScanNameUnknown):
            scan_jobs.remove_job("Help")


class TestGetJobs:

    def test_when_no_jobs(self, scan_jobs):
        assert scan_jobs.get_jobs() == []

    def test_get_all_jobs(self, scan_jobs):
        job1 = [42]
        job2 = {6: 7}
        scan_jobs.add_job('a', job1)
        scan_jobs.add_job('b', job2)
        assert job1 in scan_jobs.get_jobs()
        assert job2 in scan_jobs.get_jobs()
        assert len(scan_jobs.get_jobs()) == 2


class TestGetJobIds:

    def test_has_the_ids(self, scan_jobs):
        job1 = [42]
        job2 = {6: 7}
        scan_jobs.add_job('a', job1)
        scan_jobs.add_job('b', job2)
        assert set(scan_jobs.get_job_ids()) == set(['a', 'b'])


class TestExistsJobWith:

    def test_reports_true_for_inserted(self, scan_jobs):
        job = {6: 7, 3: 100, 10: 2}
        scan_jobs.add_job('a', job)
        for key in job:
            assert scan_jobs.exists_job_with(key, job[key])

    def test_reports_false_for_unknown(self, scan_jobs):
        job = {6: 7, 3: 100, 10: 2}
        scan_jobs.add_job('a', job)
        assert scan_jobs.exists_job_with(6, 'help') is False
        assert scan_jobs.exists_job_with('help', 6) is False
