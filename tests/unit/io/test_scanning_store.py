from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest
from scanomatic.io.scanning_store import (
    ScanningStore, ScanJobCollisionError, ScanJobUnknownError, ScanJob, Scanner
)


@pytest.fixture(scope='function')
def scanning_store():
    return ScanningStore()


JOB1 = ScanJob(
    identifier=5,
    name="Hello",
    duration=timedelta(days=1),
    interval=timedelta(minutes=20),
    scanner_id="9a8486a6f9cb11e7ac660050b68338ac",
)

JOB2 = ScanJob(
    identifier=6,
    name="Hello",
    duration=timedelta(days=1),
    interval=timedelta(minutes=20),
    scanner_id="9a8486a6f9cb11e7ac660050b68338ac",
)

SCANNER = Scanner(
    'Test',
    False,
    None,
    '9a8486a6f9cb11e7ac660050b68338ac',
)


class TestScanners:
    def test_has_test_scanner(self, scanning_store):
        assert scanning_store.has_scanner(SCANNER.identifier)

    def test_not_having_unkown_scanner(self, scanning_store):
        assert scanning_store.has_scanner("Unknown") is False

    def test_getting_scanner(self, scanning_store):
        assert scanning_store.get_scanner(SCANNER.identifier) == SCANNER

    def test_get_free(self, scanning_store):
        assert scanning_store.get_free_scanners() == [SCANNER]

    def test_get_all(self, scanning_store):
        assert scanning_store.get_all_scanners() == [SCANNER]


class TestAddJob:
    def test_add_jobb(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        assert JOB1.identifier in scanning_store.get_scanjob_ids()

    def test_add_duplicate_job_raises(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        with pytest.raises(ScanJobCollisionError):
            scanning_store.add_scanjob(JOB1)


class TestRemoveJob:
    def test_remove_job(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        scanning_store.remove_scanjob(JOB1.identifier)
        assert JOB1 not in scanning_store.get_all_scanjobs()

    def test_remove_unknown_job_raises(self, scanning_store):
        with pytest.raises(ScanJobUnknownError):
            scanning_store.remove_scanjob("Help")


class TestGetJobs:
    def test_when_no_jobs(self, scanning_store):
        assert scanning_store.get_all_scanjobs() == []
        assert scanning_store.get_scanjob_ids() == []

    def test_get_all_jobs(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        scanning_store.add_scanjob(JOB2)
        assert JOB1.identifier in scanning_store.get_scanjob_ids()
        assert JOB2.identifier in scanning_store.get_scanjob_ids()
        assert len(scanning_store.get_all_scanjobs()) == 2


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
        for key in (
            'identifier', 'name', 'duration', 'interval', 'scanner_id'
        ):
            assert scanning_store.exists_scanjob_with(key, getattr(JOB1, key))

    def test_reports_false_for_unknown(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        assert scanning_store.exists_scanjob_with(
            'identifier', 'Hello') is False


class TestGetCurrentJob:
    SCANNERID = "9a8486a6f9cb11e7ac660050b68338ac"

    @pytest.fixture
    def store(self, scanning_store):
        scanning_store.add_scanjob(ScanJob(
            identifier='1',
            name='Foo',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID
        ))
        scanning_store.add_scanjob(ScanJob(
            identifier='2',
            name='Bar',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID,
            start=datetime(1985, 10, 26, 1, 20)
        ))
        scanning_store.add_scanjob(ScanJob(
            identifier='3',
            name='Baz',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID,
            start=datetime(1985, 10, 26, 1, 35)
        ))
        scanning_store.add_scanjob(ScanJob(
            identifier='4',
            name='Biz',
            duration=timedelta(minutes=30),
            interval=timedelta(seconds=5),
            scanner_id='otherscanner',
            start=datetime(1985, 10, 26, 1, 20)
        ))
        return scanning_store

    @pytest.mark.parametrize('t, jobname', [
        (datetime(1985, 10, 26, 1, 20), 'Bar'),
        (datetime(1985, 10, 26, 1, 35), 'Baz'),
    ])
    def test_has_active_job(self, store, t, jobname):
        job = store.get_current_scanjob(self.SCANNERID, t)
        assert job is not None and job.name == jobname

    @pytest.mark.parametrize('t', [
        datetime(1985, 10, 26, 1, 15),
        datetime(1985, 10, 26, 1, 25),
        datetime(1985, 10, 26, 1, 40),
    ])
    def test_no_active_job(self, store, t):
        job = store.get_current_scanjob(self.SCANNERID, t)
        assert job is None
