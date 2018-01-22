from __future__ import absolute_import
from collections import namedtuple
from datetime import datetime, timedelta

from scanomatic.util.datetime import is_utc


ScanJobBase = namedtuple(
    'ScanJobBase',
    ['identifier', 'name', 'duration', 'interval', 'scanner_id', 'start']
)


class ScanJob(ScanJobBase):
    def __new__(
        self, identifier, name, duration, interval, scanner_id, start=None,
    ):
        if start is not None:
            if not isinstance(start, datetime):
                raise ValueError('start should be a datetime')
            if not is_utc(start):
                raise ValueError('start should be UTC')
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
