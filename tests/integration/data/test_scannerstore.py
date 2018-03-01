from __future__ import absolute_import
from datetime import datetime

import pytest
from pytz import utc
import sqlalchemy as sa

from scanomatic.data.scannerstore import ScannerStore
from scanomatic.models.scanner import Scanner


@pytest.fixture
def store(dbconnection, dbmetadata):
    return ScannerStore(dbconnection, dbmetadata)


class TestAdd:
    def test_add_one(self, store, scanner01, dbconnection):
        store.add(scanner01)
        assert list(dbconnection.execute(
            'SELECT id, name, last_seen from scanners'
        )) == [(scanner01.identifier, scanner01.name, scanner01.last_seen)]

    def test_add_duplicate_id(self, store, scanner01, dbconnection):
        store.add(scanner01)
        scanner01bis = Scanner(
            identifier=scanner01.identifier,
            name='My Other First Scanner',
        )
        with pytest.raises(ScannerStore.IntegrityError):
            store.add(scanner01bis)

    def test_add_duplicate_name(self, store, scanner01, dbconnection):
        store.add(scanner01)
        scanner001 = Scanner(
            identifier='scnr001',
            name='My First Scanner',
        )
        with pytest.raises(ScannerStore.IntegrityError):
            store.add(scanner001)


@pytest.mark.usefixtures('insert_test_scanners')
class TestUpdateScannerStatus:
    def test_update(self, store, scanner01, dbconnection):
        last_seen = datetime(1985, 10, 26, 1, 24, tzinfo=utc)
        store.update_scanner_status(scanner01.identifier, last_seen=last_seen)
        assert list(dbconnection.execute(
            sa.sql.text('SELECT last_seen from scanners WHERE id = :id'),
            id=scanner01.identifier
        )) == [(last_seen, )]

    def test_update_unknown_scanner(self, store):
        dt = datetime(1985, 10, 26, 1, 24, tzinfo=utc)
        with pytest.raises(KeyError):
            store.update_scanner_status('unknown', last_seen=dt)


@pytest.mark.usefixtures('insert_test_scanners')
class TestGetScannerById:
    def test_get_existing(self, store, scanner01):
        assert store.get_scanner_by_id(scanner01.identifier) == scanner01

    def test_non_existing(self, store):
        with pytest.raises(KeyError):
            store.get_scanner_by_id('scnr33')


@pytest.mark.usefixtures('insert_test_scanners')
class TestGetAll:
    def test_get_all(self, store, scanner01, scanner02):
        assert set(store.get_all()) == {scanner01, scanner02}


@pytest.mark.usefixtures('insert_test_scanners')
class TestHasScannerWithId:
    def test_existing(self, store, scanner01):
        assert store.has_scanner_with_id(scanner01.identifier)

    def test_non_existing(self, store):
        assert not store.has_scanner_with_id('xxx')
