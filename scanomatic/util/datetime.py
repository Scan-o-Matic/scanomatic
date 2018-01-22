from __future__ import absolute_import
import pytz


def is_utc(dt):
    return dt.tzinfo == pytz.utc
