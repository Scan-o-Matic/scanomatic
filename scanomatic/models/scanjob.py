from __future__ import absolute_import

from collections import namedtuple
from datetime import datetime, timedelta

from scanomatic.util.datetime import is_utc

ScanJobBase = namedtuple(
    'ScanJobBase', [
        'identifier',
        'name',
        'duration',
        'interval',
        'scanner_id',
        'start_time',
        'termination_time',
        'termination_message',
    ]
)


class ScanJob(ScanJobBase):
    def __new__(
            self,
            identifier,
            name,
            duration,
            interval,
            scanner_id,
            start_time=None,
            termination_time=None,
            termination_message=None,
    ):
        if start_time is not None:
            if not isinstance(start_time, datetime):
                raise ValueError('start_time should be a datetime')
            if not is_utc(start_time):
                raise ValueError('start_time should be UTC')
        if termination_time is not None:
            if not isinstance(termination_time, datetime):
                raise ValueError('termination_time should be a datetime')
            if not is_utc(termination_time):
                raise ValueError('termination_time should be UTC')
        if not isinstance(duration, timedelta):
            raise ValueError('duration should be a timedelta')
        if not isinstance(interval, timedelta):
            raise ValueError('interval should be a timedelta')
        return super(ScanJob, self).__new__(
            self,
            identifier,
            name,
            duration,
            interval,
            scanner_id,
            start_time,
            termination_time,
            termination_message,
        )

    def is_active(self, timepoint):
        if self.start_time is None:
            return False
        if self.termination_time is None:
            end_time = self.start_time + self.duration
        else:
            end_time = self.termination_time
        return self.start_time <= timepoint <= end_time
