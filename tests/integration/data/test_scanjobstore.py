from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest
from pytz import utc

from scanomatic.data.scanjobstore import ScanJobStore
from scanomatic.data import tables
from scanomatic.models.scanjob import ScanJob

pytestmark = pytest.mark.usefixtures("insert_test_scanners")


@pytest.fixture
def store(dbconnection):
    return ScanJobStore(dbconnection)


@pytest.fixture
def scanjob01(scanner01):
    return ScanJob(
        identifier='scjb001',
        name='First scan job',
        interval=timedelta(minutes=10),
        duration=timedelta(hours=1),
        scanner_id=scanner01.identifier,
        start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc)
    )


@pytest.fixture
def scanjob02(scanner01):
    return ScanJob(
        identifier='scjb002',
        name='Second scan job',
        interval=timedelta(minutes=10),
        duration=timedelta(hours=1),
        scanner_id=scanner01.identifier,
    )


@pytest.fixture
def insert_test_scanjobs(
    dbconnection, scanjob01, scanjob02, insert_test_scanners
):
    for scanjob in [scanjob01, scanjob02]:
        dbconnection.execute(
            tables.scanjobs.insert().values(
                duration=scanjob.duration,
                id=scanjob.identifier,
                interval=scanjob.interval,
                name=scanjob.name,
                scanner_id=scanjob.scanner_id,
                start_time=scanjob.start_time,
            )
        )


class TestAddScanjob:
    def test_add_one(self, store, scanjob01, dbconnection):
        store.add_scanjob(scanjob01)
        assert (
            list(dbconnection.execute('''
                SELECT id, name, interval, duration, scanner_id, start_time
                FROM scanjobs
            ''')) == [(
                scanjob01.identifier,
                scanjob01.name,
                scanjob01.interval,
                scanjob01.duration,
                scanjob01.scanner_id,
                scanjob01.start_time,
            )]
        )

    def test_add_duplicate_id(self, store, scanjob01, scanner02, dbconnection):
        store.add_scanjob(scanjob01)
        scanjob01bis = ScanJob(
            identifier=scanjob01.identifier,
            name='First scan job bis',
            interval=timedelta(minutes=20),
            duration=timedelta(hours=2),
            scanner_id=scanner02.identifier,
        )
        with pytest.raises(ScanJobStore.IntegrityError):
            store.add_scanjob(scanjob01bis)

    def test_add_duplicate_name(
        self, store, scanjob01, scanner02, dbconnection
    ):
        store.add_scanjob(scanjob01)
        scanjob01bis = ScanJob(
            identifier='scjb001.1',
            name=scanjob01.name,
            interval=timedelta(minutes=20),
            duration=timedelta(hours=2),
            scanner_id=scanner02.identifier,
        )
        with pytest.raises(ScanJobStore.IntegrityError):
            store.add_scanjob(scanjob01bis)


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestSetStartTime:
    def test_set_start_time(self, store, scanjob02, dbconnection):
        start_time = datetime(1955, 11, 5, 6, 15, tzinfo=utc)
        store.set_scanjob_start_time(scanjob02.identifier, start_time)
        assert (
            list(dbconnection.execute(
                "SELECT start_time FROM scanjobs WHERE id = 'scjb002'",
            )) == [(start_time,)]
        )

    def test_set_conflicting_start_time(self, store, scanjob02, dbconnection):
        start_time = datetime(1985, 10, 26, 1, 21, tzinfo=utc)
        with pytest.raises(ScanJobStore.IntegrityError):
            store.set_scanjob_start_time(scanjob02.identifier, start_time)


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestHasScanJobWithName:
    def test_exists(self, store, scanjob01):
        assert store.has_scanjob_with_name(scanjob01.name)

    def test_doesnt_exist(self, store):
        assert not store.has_scanjob_with_name('unknown')


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestGetScanJobById:
    def test_get_existing(self, store, scanjob01):
        assert store.get_scanjob_by_id(scanjob01.identifier) == scanjob01

    def test_get_unknown_id(self, store):
        with pytest.raises(KeyError):
            store.get_scanjob_by_id('unknown')


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestGetAllScanJobs:
    def test_get_all(self, store, scanjob01, scanjob02):
        assert set(store.get_all_scanjobs()) == {scanjob01, scanjob02}


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestGetCurrentScanjobForScanner:
    def test_has_current_job(self, store, scanjob01):
        now = scanjob01.start_time + timedelta(minutes=1)
        assert (
            store.get_current_scanjob_for_scanner(scanjob01.scanner_id, now)
            == scanjob01
        )

    def test_job_not_yet_started(self, store, scanjob01):
        now = scanjob01.start_time - timedelta(minutes=1)
        assert (
            store.get_current_scanjob_for_scanner(scanjob01.scanner_id, now)
            is None
        )

    def test_job_finished(self, store, scanjob01):
        now = scanjob01.start_time + scanjob01.duration + timedelta(minutes=1)
        assert (
            store.get_current_scanjob_for_scanner(scanjob01.scanner_id, now)
            is None
        )
