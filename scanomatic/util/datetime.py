import pytz


def is_utc(dt):
    return dt.tzinfo == pytz.utc
