from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest
from mock import patch

from pytz import utc

from scanomatic.io.scanning_store import (
    ScanningStore, ScanJobCollisionError, ScanJobUnknownError, Scanner,
    DuplicateIdError, UnknownIdError
)
from scanomatic.models.scanjob import ScanJob
from scanomatic.models.scan import Scan


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
    'Never On',
    False,
    None,
    '9a8486a6f9cb11e7ac660050b68338ac',
)

SCANNER_POWER = Scanner(
    'Always On',
    True,
    None,
    '350986224086888954',
)


class TestScanners:
    def test_has_test_scanner(self, scanning_store):
        assert scanning_store.has_scanner(SCANNER.identifier)

    def test_not_having_unkown_scanner(self, scanning_store):
        assert scanning_store.has_scanner("Unknown") is False

    @pytest.mark.parametrize('scanner', (SCANNER, SCANNER_POWER))
    def test_getting_scanner(self, scanning_store, scanner):
        assert scanning_store.get_scanner(scanner.identifier) == scanner

    def test_get_free(self, scanning_store):
        assert set(scanning_store.get_free_scanners()) == {
            SCANNER, SCANNER_POWER,
        }

    def test_get_all(self, scanning_store):
        assert set(scanning_store.get_all_scanners()) == {
            SCANNER, SCANNER_POWER,
        }

    @patch.dict('os.environ', {'SOM_HIDE_TEST_SCANNERS': '1'})
    def test_has_no_scanner_if_not_using_test_scanners(self):
        """Ensures no test scanners show up in live systems.

        Note that we can't use the `scanning_store` fixture here since
        then the patch doesn't work.
        """
        assert ScanningStore().get_all_scanners() == []


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


class TestGetScanjob:
    def test_existing_job(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        assert scanning_store.get_scanjob(JOB1.identifier) == JOB1

    def test_unknown_job(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        with pytest.raises(ScanJobUnknownError):
            scanning_store.get_scanjob('unknown')


class TestUpdateScanjob:
    def test_update_existing(self, scanning_store):
        scanning_store.add_scanjob(JOB1)
        updated_scanjob = ScanJob(
            identifier=JOB1.identifier,
            name="Bye",
            duration=JOB1.duration,
            interval=JOB1.interval,
            scanner_id=JOB1.scanner_id,
        )
        scanning_store.update_scanjob(updated_scanjob)
        assert scanning_store.get_scanjob(JOB1.identifier) == updated_scanjob

    def test_update_unknown(self, scanning_store):
        with pytest.raises(ScanJobUnknownError):
            scanning_store.update_scanjob(JOB1)


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


class TestCurrentScanJob:
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
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc)
        ))
        scanning_store.add_scanjob(ScanJob(
            identifier='3',
            name='Baz',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID,
            start_time=datetime(1985, 10, 26, 1, 35, tzinfo=utc)
        ))
        scanning_store.add_scanjob(ScanJob(
            identifier='4',
            name='Biz',
            duration=timedelta(minutes=30),
            interval=timedelta(seconds=5),
            scanner_id='otherscanner',
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc)
        ))
        return scanning_store

    @pytest.mark.parametrize('t, jobname', [
        (datetime(1985, 10, 26, 1, 20, tzinfo=utc), 'Bar'),
        (datetime(1985, 10, 26, 1, 35, tzinfo=utc), 'Baz'),
    ])
    def test_get_current_scanjob_with_active_job(self, store, t, jobname):
        job = store.get_current_scanjob(self.SCANNERID, t)
        assert job is not None and job.name == jobname

    @pytest.mark.parametrize('t', [
        datetime(1985, 10, 26, 1, 15, tzinfo=utc),
        datetime(1985, 10, 26, 1, 25, tzinfo=utc),
        datetime(1985, 10, 26, 1, 40, tzinfo=utc),
    ])
    def test_get_current_scanjob_with_no_active_job(self, store, t):
        job = store.get_current_scanjob(self.SCANNERID, t)
        assert job is None

    @pytest.mark.parametrize('t, expected', [
        (datetime(1985, 10, 26, 1, 15, tzinfo=utc), False),
        (datetime(1985, 10, 26, 1, 20, tzinfo=utc), True),
        (datetime(1985, 10, 26, 1, 25, tzinfo=utc), False),
        (datetime(1985, 10, 26, 1, 35, tzinfo=utc), True),
        (datetime(1985, 10, 26, 1, 40, tzinfo=utc), False),
    ])
    def test_has_current_scanjob(self, store, t, expected):
        assert store.has_current_scanjob(self.SCANNERID, t) is expected
