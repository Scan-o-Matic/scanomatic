from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest
from pytz import utc

from scanomatic.data.scanjobstore import ScanJobStore
from scanomatic.models.scanjob import ScanJob

pytestmark = pytest.mark.usefixtures("insert_test_scanners")


@pytest.fixture
def store(dbconnection, dbmetadata):
    return ScanJobStore(dbconnection, dbmetadata)


class TestAddScanjob:

    def test_add_one(self, store, scanjob01, dbconnection):
        store.add_scanjob(
            ScanJob(
                identifier='scjb01',
                name='Test Scan Job',
                interval=timedelta(minutes=5),
                duration=timedelta(minutes=20),
                scanner_id='scnr01',
                start_time=datetime(
                    1985, 10, 26, 1, 20, tzinfo=utc
                ),
                termination_time=datetime(
                    1985, 10, 26, 1, 21, tzinfo=utc
                ),
                termination_message='Stooop!',
            )
        )
        assert (
            list(
                dbconnection.execute(
                    '''
                SELECT id, name, interval, duration, scanner_id, start_time,
                       termination_time, termination_message
                FROM scanjobs
            '''
                )
            ) == [(
                'scjb01',
                'Test Scan Job',
                timedelta(minutes=5),
                timedelta(minutes=20),
                'scnr01',
                datetime(
                    1985, 10, 26, 1, 20, tzinfo=utc
                ),
                datetime(
                    1985, 10, 26, 1, 21, tzinfo=utc
                ),
                'Stooop!',
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

    def test_no_conflict_with_terminated_scanjob(self, store, dbconnection):
        dbconnection.execute(
            '''
            INSERT INTO scanners(id) values ('scanner001');
            INSERT INTO scanjobs(id, scanner_id, name, duration, interval,
                                 start_time, termination_time)
                   VALUES ('scnjb001', 'scanner001', 'Test Terminated Scanjob',
                           '5 minutes', '1 minute', '1985-10-26 01:20:00+00',
                           '1985-10-26 01:21:00+00');
            INSERT INTO scanjobs(id, scanner_id, name, duration, interval,
                                 start_time)
                   VALUES ('scnjb002', 'scanner001', 'Test Second Scanjob',
                           '5 minutes', '1 minute', NULL);
            '''
        )
        start_time = datetime(1985, 10, 26, 1, 22, tzinfo=utc)
        store.set_scanjob_start_time('scnjb002', start_time)


class TestTerminateScanjob:

    def test_started_scanjob(self, store, dbconnection):
        dbconnection.execute(
            '''
            INSERT INTO scanners(id) values ('scanner001');
            INSERT INTO scanjobs(id, scanner_id, name, duration, interval,
                                 start_time)
            VALUES ('scnjb001', 'scanner001', 'Test Scanjob', '5 minutes',
                    '1 minute', '1985-10-26 01:20:00+00');
            '''
        )
        termination_time = datetime(1985, 10, 26, 1, 21, tzinfo=utc)
        store.terminate_scanjob('scnjb001', termination_time, "Just 'cause")
        assert list(
            dbconnection.execute(
                ''' SELECT termination_time, termination_message
                FROM scanjobs WHERE id = 'scnjb001'
            '''
            )
        ) == [(termination_time, "Just 'cause")]

    def test_unknown_scanjob(self, store, dbconnection):
        termination_time = datetime(1985, 10, 26, 1, 21, tzinfo=utc)
        with pytest.raises(LookupError):
            store.terminate_scanjob('unknown', termination_time, "Just 'cause")

    def test_not_started_scanjob(self, store, dbconnection):
        dbconnection.execute(
            '''
            INSERT INTO scanners(id) values ('scanner001');
            INSERT INTO scanjobs(id, scanner_id, name, duration, interval,
                                 start_time)
            VALUES ('scnjb001', 'scanner001', 'Test Scanjob', '5 minutes',
                    '1 minute', NULL);
            '''
        )
        termination_time = datetime(1985, 10, 26, 1, 21, tzinfo=utc)
        with pytest.raises(store.IntegrityError):
            store.terminate_scanjob('scnjb001', termination_time, "Message")


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

    def test_get_terminated(self, store, dbconnection):
        dbconnection.execute(
            '''
            INSERT INTO scanners(id) VALUES ('scanner001');
            INSERT INTO scanjobs(id, scanner_id, name, duration, interval,
                                 start_time,
                                 termination_time, termination_message)
                   VALUES ('scnjb001', 'scanner001', 'Test Scanjob',
                           '5 minutes', '1 minute', '1985-10-26 01:20:00+00',
                           '1985-10-26 01:21:00+00', 'Bla bli blu');
            '''
        )
        scanjob = store.get_scanjob_by_id('scnjb001')
        assert scanjob.termination_time == datetime(
            1985, 10, 26, 1, 21, tzinfo=utc
        )
        assert scanjob.termination_message == 'Bla bli blu'


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestGetAllScanJobs:
    def test_get_all(self, store, scanjob01, scanjob02):
        assert set(store.get_all_scanjobs()) == {scanjob01, scanjob02}


@pytest.mark.usefixtures('insert_test_scanjobs')
class TestGetCurrentScanjobForScanner:
    def test_has_current_job(self, store, scanjob01):
        now = scanjob01.start_time + timedelta(minutes=1)
        assert (
            store.get_current_scanjob_for_scanner(scanjob01.scanner_id,
                                                  now) == scanjob01
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
            store.get_current_scanjob_for_scanner(scanjob01.scanner_id, now) is
            None
        )

    def test_job_terminated(self, store, dbconnection):
        dbconnection.execute(
            '''
            INSERT INTO scanners(id) values ('scanner001');
            INSERT INTO scanjobs(id, scanner_id, name, duration, interval,
                                 start_time, termination_time)
            VALUES ('scnjb001', 'scanner001', 'Test Scanjob', '5 minutes',
                    '1 minute', '1985-10-26 01:20:00+00',
                    '1985-10-26 01:21:00+00');
            '''
        )
        now = datetime(1985, 10, 26, 1, 22, tzinfo=utc)
        assert (
            store.get_current_scanjob_for_scanner('scanner001', now) is None
        )
