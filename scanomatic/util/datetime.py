from __future__ import absolute_import
from datetime import datetime, timedelta

import pytz


def is_utc(dt):
    return (
        dt.tzinfo is not None
        and dt.tzinfo.utcoffset(dt) == dt.tzinfo.dst(dt) == timedelta(0)
    )


def timestamp(dt):
    epoch = datetime(1970, 1, 1, tzinfo=pytz.utc)
    return (dt - epoch).total_seconds()
