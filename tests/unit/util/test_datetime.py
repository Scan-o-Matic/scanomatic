from __future__ import absolute_import
from datetime import datetime, timedelta, tzinfo

import pytest
from pytz import utc, timezone

from scanomatic.util.datetime import is_utc


class CustomUTC(tzinfo):
    def utcoffset(self, _dt):
        return timedelta(0)

    def tzname(self, _dt):
        return 'UTC'

    def dst(self, _dt):
        return timedelta(0)


class EternalAtlanticSummer(tzinfo):
    def utcoffset(self, _dt):
        return timedelta(0)

    def tzname(self, _dt):
        return 'UTC'

    def dst(self, _dt):
        return timedelta(hours=1)


class TestIsUtc:
    @pytest.mark.parametrize('date', [
        datetime(1985, 10, 26, 1, 20, tzinfo=utc),
        datetime(1985, 10, 26, 1, 20, tzinfo=CustomUTC()),
    ])
    def test_utc(self, date):
        assert is_utc(date)

    @pytest.mark.parametrize('date', [
        datetime(1985, 10, 26, 1, 20),
        timezone('America/Scoresbysund').localize(datetime(1985, 7, 26, 1, 20))
    ])
    def test_not_utc(self, date):
        assert not is_utc(date)
