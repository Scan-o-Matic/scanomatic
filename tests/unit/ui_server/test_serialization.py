from __future__ import absolute_import
import datetime as dt
import pytz

from scanomatic.ui_server.serialization import job2json
from scanomatic.models.scanjob import ScanJob


def test_serialize_job_without_starttime():
    job = ScanJob(
        identifier='test',
        name='testing',
        duration=dt.timedelta(seconds=15),
        interval=dt.timedelta(hours=22),
        scanner_id='hohoho',
        start_time=None,
    )

    assert job2json(job) == {
        'identifier': job.identifier,
        'name': job.name,
        'duration': job.duration.total_seconds(),
        'interval': job.interval.total_seconds(),
        'scannerId': job.scanner_id,
        'startTime': None,
    }


def test_serialize_job_with_starttime():
    job = ScanJob(
        identifier='test',
        name='testing',
        duration=dt.timedelta(seconds=15),
        interval=dt.timedelta(hours=22),
        scanner_id='hohoho',
        start_time=dt.datetime(
            1967,
            9,
            3,
            hour=5,
            tzinfo=pytz.timezone('Europe/Stockholm'),
        ).astimezone(pytz.utc),
    )

    assert job2json(job) == {
        'identifier': job.identifier,
        'name': job.name,
        'duration': job.duration.total_seconds(),
        'interval': job.interval.total_seconds(),
        'scannerId': job.scanner_id,
        'startTime': '1967-09-03T04:00:00Z',
    }
