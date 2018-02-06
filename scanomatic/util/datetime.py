from __future__ import absolute_import
from datetime import timedelta

import pytz


def is_utc(dt):
    return (
        dt.tzinfo is not None
        and dt.tzinfo.utcoffset(dt) == dt.tzinfo.dst(dt) == timedelta(0)
    )
