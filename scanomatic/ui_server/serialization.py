from __future__ import absolute_import
from scanomatic.util.datetime import is_utc


DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def job2json(job):
    obj = {
        'identifier': job.identifier,
        'name': job.name,
        'duration': job.duration.total_seconds(),
        'interval': job.interval.total_seconds(),
        'scannerId': job.scanner_id,
    }
    if job.start_time is not None:
        assert is_utc(job.start_time)
        obj['startTime'] = job.start_time.strftime(DATETIME_FORMAT)
    return obj


def scan2json(scan):
    return {
        'id': scan.id,
        'startTime': scan.start_time.strftime(DATETIME_FORMAT),
        'endTime': scan.end_time.strftime(DATETIME_FORMAT),
        'scanJobId': scan.scanjob_id,
        'digest': scan.digest,
    }


def scanner_status2json(status):
    obj = {
        'job': status.job,
    }
    if status.server_time is not None:
        assert is_utc(status.server_time)
        obj['serverTime'] = status.server_time.strftime(DATETIME_FORMAT)
    return obj


def scanner2json(scanner, power=False, owner=None):
    obj = scanner._asdict()
    obj['power'] = power
    obj['owner'] = owner
    return obj
