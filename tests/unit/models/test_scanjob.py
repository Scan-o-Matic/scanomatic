from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest

from scanomatic.models.scanjob import ScanJob


class TestInit:
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


class TestIsActive:
    def test_not_active_if_not_started(self):
        job = ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(days=3),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
        )
        assert not job.is_active(datetime(1985, 10, 26, 1, 20))

    @pytest.fixture
    def started_job(self):
        return ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id='yyyy',
            start=datetime(1985, 10, 26, 1, 20)
        )

    @pytest.mark.parametrize('now', [
        datetime(1985, 10, 26, 1, 20),
        datetime(1985, 10, 26, 1, 20, 30),
        datetime(1985, 10, 26, 1, 21),
    ])
    def test_active(self, started_job, now):
        assert started_job.is_active(now)

    @pytest.mark.parametrize('now', [
        datetime(1985, 10, 26, 1, 19),
        datetime(1985, 10, 26, 1, 22),
    ])
    def test_not_active(self, started_job, now):
        assert not started_job.is_active(now)
