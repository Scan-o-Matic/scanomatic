from collections import namedtuple
from datetime import datetime


ScanJobBase = namedtuple(
    'ScanJobBase',
    ['identifier', 'name', 'duration', 'interval', 'scanner_id', 'start']
)


class ScanJob(ScanJobBase):
    def __new__(
        self, identifier, name, duration, interval, scanner_id, start=None,
    ):
        if start is not None and not isinstance(start, datetime):
            raise ValueError('start should be a datetime')
        return super(ScanJob, self).__new__(
            self, identifier, name, duration, interval, scanner_id, start,
        )
    pass
