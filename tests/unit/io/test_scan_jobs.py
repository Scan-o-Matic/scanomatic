from __future__ import absolute_import

import pytest
from scanomatic.io.scan_jobs import (
    ScanJobs, ScanJobCollisionError, ScanJobUnknownError, ScanJob
)


@pytest.fixture(scope='function')
def scan_jobs():
    return ScanJobs()


JOB1 = ScanJob(
    identifier=5,
    name="Hello",
    duration="to",
    interval="you",
    scanner="!",
)

JOB2 = ScanJob(
    identifier=6,
    name="Hello",
    duration="to",
    interval="you",
    scanner="!",
)


class TestAddJob:
    def test_add_jobb(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        assert JOB1 in scan_jobs.get_jobs()

    def test_add_duplicate_job_raises(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        with pytest.raises(ScanJobCollisionError):
            scan_jobs.add_job(JOB1)


class TestRemoveJob:
    def test_remove_job(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        scan_jobs.remove_job(JOB1.identifier)
        assert JOB1 not in scan_jobs.get_jobs()

    def test_remove_unknown_job_raises(self, scan_jobs):
        with pytest.raises(ScanJobUnknownError):
            scan_jobs.remove_job("Help")


class TestGetJobs:
    def test_when_no_jobs(self, scan_jobs):
        assert scan_jobs.get_jobs() == []

    def test_get_all_jobs(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        scan_jobs.add_job(JOB2)
        assert JOB1 in scan_jobs.get_jobs()
        assert JOB2 in scan_jobs.get_jobs()
        assert len(scan_jobs.get_jobs()) == 2


class TestGetJobIds:
    def test_has_the_ids(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        scan_jobs.add_job(JOB2)
        assert set(scan_jobs.get_job_ids()) == set(
            [JOB1.identifier, JOB2.identifier]
        )


class TestExistsJobWith:
    def test_reports_true_for_inserted(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        for key in ('identifier', 'name', 'duration', 'interval', 'scanner'):
            assert scan_jobs.exists_job_with(key, getattr(JOB1, key))

    def test_reports_false_for_unknown(self, scan_jobs):
        scan_jobs.add_job(JOB1)
        assert scan_jobs.exists_job_with('identifier', 'Hello') is False
