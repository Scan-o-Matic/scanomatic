from __future__ import absolute_import
import datetime as dt
import pytz

from scanomatic.models.scan import Scan
from scanomatic.models.scanjob import ScanJob
from scanomatic.ui_server.serialization import scan2json, job2json

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


class TestScan2Json(object):
    def test_simple_scan(self):
        scan = Scan(
            id='xxxx',
            scanjob_id='yyyy',
            start_time=dt.datetime(1985, 10, 26, 1, 20, tzinfo=pytz.utc),
            end_time=dt.datetime(1985, 10, 26, 1, 21, tzinfo=pytz.utc),
            digest='foo:bar',
        )
        assert scan2json(scan) == {
            'id': 'xxxx',
            'digest': 'foo:bar',
            'startTime': '1985-10-26T01:20:00Z',
            'endTime': '1985-10-26T01:21:00Z',
            'scanJobId': 'yyyy',
        }
