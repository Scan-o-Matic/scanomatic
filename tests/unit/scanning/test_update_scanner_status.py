from __future__ import absolute_import
from datetime import datetime
from uuid import uuid4

from freezegun import freeze_time
import mock
from prometheus_client import REGISTRY
import pytest
from pytz import utc

from scanomatic.scanning.update_scanner_status import update_scanner_status


@pytest.fixture
def scanner():
    return uuid4().hex


@pytest.fixture
def update():
    return {
        'job': 'job007',
        'images_to_send': 5,
        'next_scheduled_scan': datetime(1985, 10, 26, 1, 35, tzinfo=utc),
        'start_time': datetime(1985, 10, 26, 1, 20, tzinfo=utc),
        'devices': ['epson'],
    }


@pytest.fixture
def db():
    return mock.MagicMock()


class TestMetrics:
    @pytest.mark.parametrize('job, value', [('job001', 1), (None, 0)])
    def test_scanner_current_jobs(self, db, scanner, update, job, value):
        update['job'] = job
        update_scanner_status(db, scanner, **update)
        assert REGISTRY.get_sample_value(
            'scanner_current_jobs', labels={'scanner': scanner}) == value

    @pytest.mark.parametrize('metric, value', [
        ('scanner_queued_uploads', 5),
        ('scanner_start_time_seconds', 499137600.0),
        ('scanner_last_status_update_time_seconds', 499137660.0),
        ('scanner_status_updates_total', 1),
    ])
    def test_other_metrics(self, db, scanner, update, metric, value):
        with freeze_time(datetime(1985, 10, 26, 1, 21, tzinfo=utc)):
            update_scanner_status(db, scanner, **update)
        assert REGISTRY.get_sample_value(
            metric, labels={'scanner': scanner}) == value
