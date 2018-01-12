from __future__ import absolute_import

import pytest
from scanomatic.io.scanning_store import (
    ScanningStore, ScanJobCollisionError, ScanJobUnknownError, ScanJob
)


@pytest.fixture(scope='function')
def scanning_store():
    return ScanningStore()


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


class TestScanners:
    def test_has_test_scanner(self, scanning_store):
        assert scanning_store.has_scanner("Test")

    def test_not_having_unkown_scanner(self, scanning_store):
        assert scanning_store.has_scanner("Unknown") is False

    def test_getting_scanner(self, scanning_store):
        assert scanning_store.get_scanner("Test") == {
            "name": "Test",
            "power": False,
            "owner": None,
        }

    def test_get_free(self, scanning_store):
        assert scanning_store.get_free_scanners() == [
            {
                "name": "Test",
                "power": False,
                "owner": None,
            },
        ]

    def test_get_all(self, scanning_store):
        assert scanning_store.get_all_scanners() == [
            {
                "name": "Test",
                "power": False,
                "owner": None,
            },
        ]


class TestAddJob:
    def test_add_jobb(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        assert JOB1 in scanning_store.get_scanjobs()

    def test_add_duplicate_job_raises(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        with pytest.raises(ScanJobCollisionError):
            scanning_store.add_scanjob(JOB1)


class TestRemoveJob:
    def test_remove_job(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        scanning_store.remove_scanjob(JOB1.identifier)
        assert JOB1 not in scanning_store.get_scanjobs()

    def test_remove_unknown_job_raises(self, scanning_store):
        with pytest.raises(ScanJobUnknownError):
            scanning_store.remove_scanjob("Help")


class TestGetJobs:
    def test_when_no_jobs(self, scanning_store):
        assert scanning_store.get_scanjobs() == []

    def test_get_all_jobs(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        scanning_store.add_scanjob(JOB2)
        assert JOB1 in scanning_store.get_scanjobs()
        assert JOB2 in scanning_store.get_scanjobs()
        assert len(scanning_store.get_scanjobs()) == 2


class TestGetJobIds:
    def test_has_the_ids(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        scanning_store.add_scanjob(JOB2)
        assert set(scanning_store.get_scanjob_ids()) == set(
            [JOB1.identifier, JOB2.identifier]
        )


class TestExistsJobWith:
    def test_reports_true_for_inserted(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        for key in ('identifier', 'name', 'duration', 'interval', 'scanner'):
            assert scanning_store.exists_scanjob_with(key, getattr(JOB1, key))

    def test_reports_false_for_unknown(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        assert scanning_store.exists_scanjob_with('identifier', 'Hello') is False
