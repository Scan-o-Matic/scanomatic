from __future__ import absolute_import
from collections import namedtuple
from datetime import datetime, timedelta


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
        if not isinstance(duration, timedelta):
            raise ValueError('duration should be a timedelta')
        if not isinstance(interval, timedelta):
            raise ValueError('interval should be a timedelta')
        return super(ScanJob, self).__new__(
            self, identifier, name, duration, interval, scanner_id, start,
        )

    def is_active(self, timepoint):
        return (
            self.start is not None
            and self.start <= timepoint <= self.start + self.duration
        )
