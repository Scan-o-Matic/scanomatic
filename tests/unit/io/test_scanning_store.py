from __future__ import absolute_import
from datetime import datetime

import pytest

from pytz import utc

from scanomatic.io.scanning_store import ScanningStore
from scanomatic.models.scanner import Scanner
from scanomatic.models.scannerstatus import ScannerStatus


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
