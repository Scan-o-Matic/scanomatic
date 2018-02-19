from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest

from pytz import utc

from scanomatic.io.scanning_store import (
    ScanningStore,
    DuplicateIdError, DuplicateNameError, UnknownIdError
)
from scanomatic.models.scanjob import ScanJob
from scanomatic.models.scan import Scan
from scanomatic.models.scanner import Scanner
from scanomatic.models.scannerstatus import ScannerStatus


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

SCANNER_ONE = Scanner(
    'Scanner one',
    '9a8486a6f9cb11e7ac660050b68338ac',
)

SCANNER_TWO = Scanner(
    'Scanner two',
    '350986224086888954',
)


@pytest.fixture(scope='function')
def scanning_store():
    store = ScanningStore()
    store.add(SCANNER_ONE)
    store.add(SCANNER_TWO)
    return store


class TestScanners:
    def test_has_test_scanner(self, scanning_store):
        assert scanning_store.exists(Scanner, SCANNER_ONE.identifier)

    def test_not_having_unknown_scanner(self, scanning_store):
        assert scanning_store.exists(Scanner, "Unknown") is False

    @pytest.mark.parametrize('scanner', (SCANNER_ONE, SCANNER_TWO))
    def test_getting_scanner(self, scanning_store, scanner):
        assert scanning_store.get(Scanner, scanner.identifier) == scanner

    def test_add_scanner(self, scanning_store):
        scanner = Scanner("Deep Thought", "42")
        scanning_store.add(scanner)
        assert set(scanning_store.find(Scanner)) == {
            scanner, SCANNER_ONE, SCANNER_TWO}

    def test_no_add_scanner_duplicate_id(self, scanning_store):
        with pytest.raises(DuplicateIdError):
            scanning_store.add(SCANNER_ONE)

    def test_no_add_scanner_duplicate_name(self, scanning_store):
        scanner = Scanner("Scanner two", "2")
        with pytest.raises(DuplicateNameError):
            scanning_store.add(scanner)

    def test_get_scanner(self, scanning_store):
        assert (
            scanning_store.get(Scanner, SCANNER_ONE.identifier) == SCANNER_ONE
        )

    def test_no_get_scanner_by_unknown_id(self, scanning_store):
        with pytest.raises(UnknownIdError):
            assert scanning_store.get(Scanner, "42")


class TestScannerStatus:
    def test_get_scanner_status_list(self, scanning_store):
        assert scanning_store.get_scanner_status_list(
            SCANNER_ONE.identifier) == []

    def test_no_get_scanner_status_list_unknown(self, scanning_store):
        with pytest.raises(UnknownIdError):
            scanning_store.get_scanner_status_list("42")

    def test_get_lastest_scanner_status_empty(self, scanning_store):
        assert scanning_store.get_latest_scanner_status(
            '9a8486a6f9cb11e7ac660050b68338ac') is None

    def test_get_lastest_scanner_status(self, scanning_store):
        status1 = ScannerStatus(
            job="j0bid",
            server_time=datetime.now(utc),
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
            images_to_send=3,
            next_scheduled_scan=datetime(1985, 10, 26, 1, 21, tzinfo=utc),
            devices=['epson'],
        )
        status2 = status1._replace(server_time=datetime.now(utc))
        scanning_store.add_scanner_status(SCANNER_ONE.identifier, status1)
        scanning_store.add_scanner_status(SCANNER_ONE.identifier, status2)
        assert scanning_store.get_latest_scanner_status(
            SCANNER_ONE.identifier) == status2

    def test_add_scanner_status(self, scanning_store):
        status = ScannerStatus(
            job="j0bid",
            server_time=datetime.now(utc),
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
            images_to_send=3,
            next_scheduled_scan=datetime(1985, 10, 26, 1, 21, tzinfo=utc),
            devices=['epson'],
        )
        scanning_store.add_scanner_status(
            '9a8486a6f9cb11e7ac660050b68338ac', status)
        assert scanning_store.get_latest_scanner_status(
            '9a8486a6f9cb11e7ac660050b68338ac') == status


class TestAddJob:
    def test_add_jobb(self, scanning_store):
        scanning_store.add(JOB1)
        assert scanning_store.exists(ScanJob, JOB1.identifier)

    def test_add_duplicate_job_raises(self, scanning_store):
        scanning_store.add(JOB1)
        with pytest.raises(DuplicateIdError):
            scanning_store.add(JOB1)


class TestGetJobs:
    def test_when_no_jobs(self, scanning_store):
        assert list(scanning_store.find(ScanJob)) == []

    def test_get_all_jobs(self, scanning_store):
        scanning_store.add(JOB1)
        scanning_store.add(JOB2)
        assert set(scanning_store.find(ScanJob)) == {JOB1, JOB2}


class TestGetScanjob:
    def test_existing_job(self, scanning_store):
        scanning_store.add(JOB1)
        assert scanning_store.get(ScanJob, JOB1.identifier) == JOB1

    def test_unknown_job(self, scanning_store):
        scanning_store.add(JOB1)
        with pytest.raises(UnknownIdError):
            scanning_store.get(ScanJob, 'unknown')


class TestUpdateScanjob:
    def test_update_existing(self, scanning_store):
        scanning_store.add(JOB1)
        updated_scanjob = ScanJob(
            identifier=JOB1.identifier,
            name="Bye",
            duration=JOB1.duration,
            interval=JOB1.interval,
            scanner_id=JOB1.scanner_id,
        )
        scanning_store.update(updated_scanjob)
        assert scanning_store.get(ScanJob, JOB1.identifier) == updated_scanjob

    def test_update_unknown(self, scanning_store):
        with pytest.raises(UnknownIdError):
            scanning_store.update(JOB1)


class TestExistsJobWith:
    def test_reports_true_for_inserted(self, scanning_store):
        scanning_store.add(JOB1)
        for key in (
            'identifier', 'name', 'duration', 'interval', 'scanner_id'
        ):
            assert scanning_store.exists(ScanJob, **{key: getattr(JOB1, key)})

    def test_reports_false_for_unknown(self, scanning_store):
        scanning_store.add(JOB1)
        assert scanning_store.exists(ScanJob, identifier='Hello') is False


class TestCurrentScanJob:
    SCANNERID = "9a8486a6f9cb11e7ac660050b68338ac"

    @pytest.fixture
    def store(self, scanning_store):
        scanning_store.add(ScanJob(
            identifier='1',
            name='Foo',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID
        ))
        scanning_store.add(ScanJob(
            identifier='2',
            name='Bar',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID,
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc)
        ))
        scanning_store.add(ScanJob(
            identifier='3',
            name='Baz',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id=self.SCANNERID,
            start_time=datetime(1985, 10, 26, 1, 35, tzinfo=utc)
        ))
        scanning_store.add(ScanJob(
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


class TestScan:
    @pytest.fixture
    def scanjob1(self):
        return JOB1

    @pytest.fixture
    def scanjob2(self):
        return JOB2

    @pytest.fixture
    def scanjob1_scan1(self, scanjob1):
        return Scan(
            id='aaaa',
            scanjob_id=scanjob1.identifier,
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
            end_time=datetime(1985, 10, 26, 1, 21, tzinfo=utc),
            digest='foo:bar',
        )

    @pytest.fixture
    def scanjob1_scan2(self, scanjob1):
        return Scan(
            id='bbbb',
            scanjob_id=scanjob1.identifier,
            start_time=datetime(1985, 10, 26, 1, 30, tzinfo=utc),
            end_time=datetime(1985, 10, 26, 1, 31, tzinfo=utc),
            digest='foo:baz',
        )

    @pytest.fixture
    def scanjob2_scan1(self, scanjob2):
        return Scan(
            id='cccc',
            scanjob_id=scanjob2.identifier,
            start_time=datetime(1985, 10, 26, 1, 40, tzinfo=utc),
            end_time=datetime(1985, 10, 26, 1, 41, tzinfo=utc),
            digest='foo:baz',
        )

    def test_add_one(self, scanning_store, scanjob1, scanjob1_scan1):
        scanning_store.add(scanjob1)
        scanning_store.add(scanjob1_scan1)
        assert list(scanning_store.find(Scan)) == [scanjob1_scan1]

    def test_add_multiple(
        self, scanning_store,
        scanjob1, scanjob1_scan1, scanjob1_scan2
    ):
        scanning_store.add(scanjob1)
        scanning_store.add(scanjob1_scan1)
        scanning_store.add(scanjob1_scan2)
        assert (
            set(scanning_store.find(Scan))
            == {scanjob1_scan1, scanjob1_scan2}
        )

    def test_duplicate_id(
        self, scanning_store, scanjob1, scanjob1_scan1,
    ):
        scanning_store.add(scanjob1)
        scanning_store.add(scanjob1_scan1)
        with pytest.raises(DuplicateIdError):
            scanning_store.add(scanjob1_scan1)

    def test_add_unknown_scanjob(self, scanning_store, scanjob1_scan1):
        with pytest.raises(UnknownIdError):
            scanning_store.add(scanjob1_scan1)

    def test_get_scan(self, scanning_store, scanjob1, scanjob1_scan1):
        scanning_store.add(scanjob1)
        scanning_store.add(scanjob1_scan1)
        assert scanning_store.get(Scan, scanjob1_scan1.id) == scanjob1_scan1

    def test_get_unknown_scan(self, scanning_store):
        with pytest.raises(UnknownIdError):
            scanning_store.get(Scan, 'unknown')
