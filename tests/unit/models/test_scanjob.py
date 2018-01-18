from datetime import datetime, timedelta

import pytest

from scanomatic.models.scanjob import ScanJob


class TestScanJob:
    def test_init(self):
        job = ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(days=3),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
            start=datetime(1985, 10, 26, 1, 20),
        )
        assert job.identifier == 'xxxx'
        assert job.name == 'Unknown'
        assert job.duration == timedelta(days=3)
        assert job.interval == timedelta(minutes=5)
        assert job.scanner_id == 'yyyy'
        assert job.start == datetime(1985, 10, 26, 1, 20)

    def test_init_without_start(self):
        job = ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(days=3),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
        )
        assert job.start is None

    def test_init_bad_start(self):
        with pytest.raises(ValueError):
            ScanJob(
                identifier='xxxx',
                name='Unknown',
                duration=timedelta(days=3),
                interval=timedelta(minutes=5),
                scanner_id='yyyy',
                start='xxx',
            )
