from __future__ import absolute_import
import pytest

import sqlalchemy as sa

from scanomatic.data.scannerstore import ScannerStore
from scanomatic.data import tables
from scanomatic.models.scanner import Scanner


@pytest.fixture
def dbconnection(database):
    return sa.create_engine(database)


@pytest.fixture
def store(dbconnection):
    return ScannerStore(dbconnection)


@pytest.fixture
def scanner01():
    return Scanner(identifier='scnr01', name='My First Scanner')


@pytest.fixture
def scanner02():
    return Scanner(identifier='scnr02', name='My Second Scanner')


@pytest.fixture
def insert_test_scanners(dbconnection, scanner01, scanner02):
    for scanner in [scanner01, scanner02]:
        dbconnection.execute(
            tables.scanners.insert().values(
                name=scanner.name,
                id=scanner.identifier,
            )
        )


class TestAdd:
    def test_add_one(self, store, scanner01, dbconnection):
        store.add(scanner01)
        assert list(dbconnection.execute('SELECT id, name from scanners')) == [
            ('scnr01', 'My First Scanner'),
        ]

    def test_add_duplicate_id(self, store, scanner01, dbconnection):
        store = ScannerStore(dbconnection)
        store.add(scanner01)
        scanner01bis = Scanner(
            identifier=scanner01.identifier,
            name='My Other First Scanner',
        )
        with pytest.raises(ScannerStore.IntegrityError):
            store.add(scanner01bis)

    def test_add_duplicate_name(self, store, scanner01, dbconnection):
        store = ScannerStore(dbconnection)
        store.add(scanner01)
        scanner001 = Scanner(
            identifier='scnr001',
            name='My First Scanner',
        )
        with pytest.raises(ScannerStore.IntegrityError):
            store.add(scanner001)


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
