from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest

from pytz import utc

from scanomatic.io.scanning_store import (
    ScanningStore,
    DuplicateIdError, UnknownIdError
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


@pytest.fixture(scope='function')
def scanning_store():
    store = ScanningStore()
    return store


class TestScannerStatus:
    def test_get_scanner_status_list(self, scanning_store):
        assert scanning_store.get_scanner_status_list(
            SCANNER_ONE.identifier) == []

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
        scanning_store.add(scanjob1_scan1)
        assert list(scanning_store.find(Scan)) == [scanjob1_scan1]

    def test_add_multiple(
        self, scanning_store,
        scanjob1, scanjob1_scan1, scanjob1_scan2
    ):
        scanning_store.add(scanjob1_scan1)
        scanning_store.add(scanjob1_scan2)
        assert (
            set(scanning_store.find(Scan))
            == {scanjob1_scan1, scanjob1_scan2}
        )

    def test_duplicate_id(
        self, scanning_store, scanjob1, scanjob1_scan1,
    ):
        scanning_store.add(scanjob1_scan1)
        with pytest.raises(DuplicateIdError):
            scanning_store.add(scanjob1_scan1)

    def test_get_scan(self, scanning_store, scanjob1, scanjob1_scan1):
        scanning_store.add(scanjob1_scan1)
        assert scanning_store.get(Scan, scanjob1_scan1.id) == scanjob1_scan1

    def test_get_unknown_scan(self, scanning_store):
        with pytest.raises(UnknownIdError):
            scanning_store.get(Scan, 'unknown')
