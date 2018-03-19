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
    if job.termination_time is not None:
        obj['terminationTime'] = datetime2json(job.termination_time)
    if job.termination_message:
        obj['terminationMessage'] = job.termination_message
    return obj


def scan2json(scan):
    return {
        'id': scan.id,
        'startTime': datetime2json(scan.start_time),
        'endTime': datetime2json(scan.end_time),
        'scanJobId': scan.scanjob_id,
        'digest': scan.digest,
    }


def scanner_status2json(status):
    obj = {
        'imagesToSend': status.images_to_send,
        'serverTime': datetime2json(status.server_time),
        'startTime': datetime2json(status.start_time),
    }
    if status.next_scheduled_scan is not None:
        obj['nextScheduledScan'] = datetime2json(status.next_scheduled_scan)
    if status.job is not None:
        obj['job'] = status.job
    if status.devices is not None:
        obj['devices'] = status.devices
    return obj


def scanner2json(scanner, power=False, owner=None):
    obj = {
        'identifier': scanner.identifier,
        'name': scanner.name,
        'power': power,
    }
    if owner is not None:
        obj['owner'] = owner
    return obj
