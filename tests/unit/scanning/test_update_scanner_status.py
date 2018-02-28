from __future__ import absolute_import
from datetime import datetime
from uuid import uuid4

from freezegun import freeze_time
import mock
from prometheus_client import REGISTRY
import pytest
from pytz import utc

from scanomatic.data.scannerstore import ScannerStore
from scanomatic.scanning.update_scanner_status import (
    update_scanner_status, UpdateScannerStatusError
)


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


@pytest.fixture
def scannerstore():
    return mock.MagicMock()


def test_update_status_in_store(scannerstore, db, scanner, update):
    now = datetime(1985, 10, 26, 1, 21, tzinfo=utc)
    with freeze_time(now):
        update_scanner_status(scannerstore, db, scanner, **update)
    scannerstore.update_scanner_status.assert_called_with(scanner, last_seen=now)


def test_duplicate_scanner_name(scannerstore, db, scanner, update):
    scannerstore.has_scanner_with_id.return_value = False
    scannerstore.add.side_effect = ScannerStore.IntegrityError
    with pytest.raises(UpdateScannerStatusError):
        update_scanner_status(scannerstore, db, scanner, **update)


class TestMetrics:
    @pytest.mark.parametrize('job, value', [('job001', 1), (None, 0)])
    def test_scanner_current_jobs(
        self, scannerstore, db, scanner, update, job, value
    ):
        update['job'] = job
        update_scanner_status(scannerstore, db, scanner, **update)
        assert REGISTRY.get_sample_value(
            'scanner_current_jobs', labels={'scanner': scanner}) == value

    @pytest.mark.parametrize('metric, value', [
        ('scanner_queued_uploads', 5),
        ('scanner_start_time_seconds', 499137600.0),
        ('scanner_last_status_update_time_seconds', 499137660.0),
        ('scanner_status_updates_total', 1),
    ])
    def test_other_metrics(
        self, scannerstore, db, scanner, update, metric, value
    ):
        with freeze_time(datetime(1985, 10, 26, 1, 21, tzinfo=utc)):
            update_scanner_status(scannerstore, db, scanner, **update)
        assert REGISTRY.get_sample_value(
            metric, labels={'scanner': scanner}) == value

    @pytest.mark.parametrize('devices, value', [
        (None, 0),
        ([], 0),
        (['a'], 1),
        (['a', 'b'], 2),
    ])
    def test_current_devices(
        self, scannerstore, db, scanner, update, devices, value
    ):
        update['devices'] = devices
        update_scanner_status(scannerstore, db, scanner, **update)
        assert REGISTRY.get_sample_value(
            'scanner_current_devices', labels={'scanner': scanner}) == value
