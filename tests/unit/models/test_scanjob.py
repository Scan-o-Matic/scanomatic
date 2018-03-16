from __future__ import absolute_import
from datetime import datetime, timedelta

import pytest
from pytz import utc

from scanomatic.models.scanjob import ScanJob


class TestInit:
    def test_init(self):
        job = ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(days=3),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc),
        )
        assert job.identifier == 'xxxx'
        assert job.name == 'Unknown'
        assert job.duration == timedelta(days=3)
        assert job.interval == timedelta(minutes=5)
        assert job.scanner_id == 'yyyy'
        assert job.start_time == datetime(1985, 10, 26, 1, 20, tzinfo=utc)

    def test_init_without_start_time(self):
        job = ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(days=3),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
        )
        assert job.start_time is None

    @pytest.mark.parametrize('start_time', [
        ('xxx',),
        (datetime(1985, 10, 26, 1, 20),),
    ])
    def test_init_bad_start_time(self, start_time):
        with pytest.raises(ValueError):
            ScanJob(
                identifier='xxxx',
                name='Unknown',
                duration=timedelta(days=3),
                interval=timedelta(minutes=5),
                scanner_id='yyyy',
                start_time=start_time,
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
        assert not job.is_active(datetime(1985, 10, 26, 1, 20, tzinfo=utc))

    @pytest.fixture
    def started_job(self):
        return ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(minutes=1),
            interval=timedelta(seconds=5),
            scanner_id='yyyy',
            start_time=datetime(1985, 10, 26, 1, 20, tzinfo=utc)
        )

    @pytest.mark.parametrize('now', [
        datetime(1985, 10, 26, 1, 20, tzinfo=utc),
        datetime(1985, 10, 26, 1, 20, 30, tzinfo=utc),
        datetime(1985, 10, 26, 1, 21, tzinfo=utc),
    ])
    def test_active(self, started_job, now):
        assert started_job.is_active(now)

    @pytest.mark.parametrize('now', [
        datetime(1985, 10, 26, 1, 19, tzinfo=utc),
        datetime(1985, 10, 26, 1, 22, tzinfo=utc),
    ])
    def test_not_active(self, started_job, now):
        assert not started_job.is_active(now)

    def test_terminated_job_is_not_active(self):
        terminated_job = ScanJob(
            identifier='xxxx',
            name='Unknown',
            duration=timedelta(minutes=20),
            interval=timedelta(minutes=5),
            scanner_id='yyyy',
            start_time=datetime(
                1985, 10, 26, 1, 20, tzinfo=utc
            ),
            termination_time=datetime(
                1985, 10, 26, 1, 21, tzinfo=utc
            ),
        )
        assert not terminated_job.is_active(
            datetime(
                1985, 10, 26, 1, 22, tzinfo=utc
            )
        )
