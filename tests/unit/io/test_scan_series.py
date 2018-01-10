from __future__ import absolute_import

import pytest
from scanomatic.io import scan_series


@pytest.fixture
def remove_jobs():
    yield
    for name in scan_series.get_job_names():
        scan_series.remove_job(name)


@pytest.mark.usefixtures("remove_jobs")
class TestAddJob:

    def test_add_jobb(self):
        name = "Test"
        job = {'hello': 'to you'}
        scan_series.add_job(name, job)
        assert job in scan_series.get_jobs()

    def test_add_duplicate_job_raises(self):
        name = "Test"
        job = {'hello': 'to you'}
        scan_series.add_job(name, job)
        with pytest.raises(scan_series.ScanNameCollision):
            scan_series.add_job(name, job)


@pytest.mark.usefixtures("remove_jobs")
class TestRemoveJob:

    def test_remove_job(self):
        name = "Test"
        job = {'hello': 'to you'}
        scan_series.add_job(name, job)
        scan_series.remove_job(name)
        assert job not in scan_series.get_jobs()

    def test_remove_unknown_job_raises(self):
        with pytest.raises(scan_series.ScanNameUnknown):
            scan_series.remove_job("Help")


@pytest.mark.usefixtures("remove_jobs")
class TestGetJobs:

    def test_when_no_jobs(self):
        assert scan_series.get_jobs() == []

    def test_get_all_jobs(self):
        job1 = [42]
        job2 = {6: 7}
        scan_series.add_job('a', job1)
        scan_series.add_job('b', job2)
        assert job1 in scan_series.get_jobs()
        assert job2 in scan_series.get_jobs()
        assert len(scan_series.get_jobs()) == 2
