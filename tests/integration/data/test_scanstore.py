from __future__ import absolute_import
from datetime import datetime

import pytest
from pytz import utc

from scanomatic.data.scanstore import ScanStore
from scanomatic.models.scan import Scan


pytestmark = pytest.mark.usefixtures("insert_test_scanjobs")


@pytest.fixture
def store(dbconnection, dbmetadata):
    return ScanStore(dbconnection, dbmetadata)


class TestAdd:
    def test_add_once(self, store, scan01, dbconnection):
        store.add_scan(scan01)
        assert (
            list(dbconnection.execute('''
                SELECT id, scanjob_id, start_time, end_time, digest
                FROM scans
            ''')) == [(
                scan01.identifier,
                scan01.scanjob_id,
                scan01.start_time,
                scan01.end_time,
                scan01.digest,
            )]
        )

    def test_add_twice(self, store, scan01):
        store.add_scan(scan01)
        with pytest.raises(ScanStore.IntegrityError):
            store.add_scan(scan01)


@pytest.mark.usefixtures('insert_test_scans')
class TestGetScanById:
    def test_exists(self, store, scan01):
        assert store.get_scan_by_id(scan01.identifier) == scan01

    def test_doesnt_exist(self, store):
        with pytest.raises(KeyError):
            store.get_scan_by_id('unknown')


@pytest.mark.usefixtures('insert_test_scans')
class TestGetAllScans:
    def test_get_all(self, store, scan01, scan02):
        assert set(store.get_all_scans()) == {scan01, scan02}
