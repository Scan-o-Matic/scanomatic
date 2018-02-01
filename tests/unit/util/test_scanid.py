from datetime import datetime, timedelta

import pytest
from pytz import utc

from scanomatic.scanning import generate_scan_id
from scanomatic.models.scanjob import ScanJob


@pytest.fixture
def scanjob():
    return ScanJob(
        identifier='j00b',
        name="The Job",
        duration=timedelta(days=1),
        interval=timedelta(minutes=20),
        scanner_id="9a8486a6f9cb11e7ac660050b68338ac",
        start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
    )


@pytest.mark.parametrize('delay, scanid', [
    (timedelta(minutes=0), 'j00b_0000_0.0000'),
    (timedelta(minutes=10), 'j00b_0001_600.0000'),
    (timedelta(minutes=15), 'j00b_0001_900.0000'),
    (timedelta(minutes=20), 'j00b_0001_1200.0000'),
    (timedelta(minutes=25), 'j00b_0001_1500.0000'),
    (timedelta(minutes=30), 'j00b_0002_1800.0000'),
    (timedelta(minutes=35), 'j00b_0002_2100.0000'),
])
def test_scan_id(scanjob, delay, scanid):
    scantime = scanjob.start_time + delay
    assert generate_scan_id(scanjob, scantime) == scanid
