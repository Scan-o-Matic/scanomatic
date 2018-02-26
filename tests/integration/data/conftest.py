import pytest

import sqlalchemy as sa

from scanomatic.data import tables
from scanomatic.models.scanner import Scanner


@pytest.fixture
def dbconnection(database):
    return sa.create_engine(database)


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
