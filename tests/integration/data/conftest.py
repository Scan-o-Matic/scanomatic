from datetime import datetime, timedelta

import pytest
from pytz import utc
import sqlalchemy as sa

from scanomatic.data import tables
from scanomatic.models.scan import Scan
from scanomatic.models.scanjob import ScanJob
from scanomatic.models.scanner import Scanner


@pytest.fixture
def dbconnection(database):
    return sa.create_engine(database)


@pytest.fixture
def dbmetadata(dbconnection):
    meta = sa.MetaData()
    meta.reflect(bind=dbconnection)
    return meta


@pytest.fixture
def scanner01():
    return Scanner(identifier='scnr01', name='My First Scanner')


@pytest.fixture
def scanner02():
    return Scanner(identifier='scnr02', name='My Second Scanner')


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
def scan01(scanjob01):
    return Scan(
        id='sc001',
        start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
        end_time=datetime(1985, 10, 26, 1, 21, tzinfo=utc),
        scanjob_id=scanjob01.identifier,
        digest='sha256:abcdef123456',
    )


@pytest.fixture
def scan02(scanjob01):
    return Scan(
        id='sc002',
        start_time=datetime(1985, 10, 26, 1, 30, tzinfo=utc),
        end_time=datetime(1985, 10, 26, 1, 31, tzinfo=utc),
        scanjob_id=scanjob01.identifier,
        digest='sha256:123456abcdef',
    )


@pytest.fixture
def insert_test_scanners(dbconnection, scanner01, scanner02):
    for scanner in [scanner01, scanner02]:
        dbconnection.execute(
            tables.scanners.insert().values(
                name=scanner.name,
                id=scanner.identifier,
            )
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


@pytest.fixture
def insert_test_scans(
    dbconnection, dbmetadata, scan01, scan02, insert_test_scanjobs,
):
    for scan in [scan01, scan02]:
        dbconnection.execute(
            dbmetadata.tables['scans'].insert().values(
                id=scan.identifier,
                start_time=scan.start_time,
                end_time=scan.end_time,
                digest=scan.digest,
                scanjob_id=scan.scanjob_id,
            )
        )
