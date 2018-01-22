from scanomatic.util.datetime import is_utc


def job2json(job):
    obj = {
        'identifier': job.identifier,
        'name': job.name,
        'duration': job.duration.total_seconds(),
        'interval': job.interval.total_seconds(),
        'scannerId': job.scanner_id,
    }
    if job.start is not None:
        assert is_utc(job.start)
        obj['start'] = job.start.strftime('%Y-%m-%dT%H:%M:%SZ')
    return obj
