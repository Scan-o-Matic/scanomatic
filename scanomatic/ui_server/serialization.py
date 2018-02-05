from __future__ import absolute_import
from scanomatic.util.datetime import is_utc


DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def datetime2json(dt):
    assert is_utc(dt), '{} is not UTC'.format(dt)
    return dt.strftime(DATETIME_FORMAT)


def job2json(job):
    obj = {
        'identifier': job.identifier,
        'name': job.name,
        'duration': job.duration.total_seconds(),
        'interval': job.interval.total_seconds(),
        'scannerId': job.scanner_id,
    }
    if job.start_time is not None:
        obj['startTime'] = datetime2json(job.start_time)
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
        'imagesToSend': status.images_to_send,
        'job': status.job,
        'serverTime': datetime2json(status.server_time),
        'startTime': datetime2json(status.start_time),
    }
    return obj


def scanner2json(scanner, power=False, owner=None):
    obj = scanner._asdict()
    obj['power'] = power
    if owner is not None:
        obj['owner'] = owner
    return obj
