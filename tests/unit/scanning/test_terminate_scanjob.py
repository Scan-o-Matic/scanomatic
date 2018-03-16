from datetime import datetime, timedelta

from freezegun import freeze_time
from mock import MagicMock
import pytest
from pytz import utc

from scanomatic.data.scanjobstore import ScanJobStore
from scanomatic.models.scanjob import ScanJob
from scanomatic.scanning.terminate_scanjob import (
    TerminateScanJobError, UnknownScanjobError, terminate_scanjob
)


def make_scanjob(
        start_time=datetime(
            1985, 10, 26, 1, 20, tzinfo=utc
        ),
        termination_time=None,
        duration=timedelta(minutes=20),
):
    return ScanJob(
        duration=duration,
        identifier='scjb000',
        interval=timedelta(minutes=5),
        name='Test Scan Job',
        scanner_id='scnr000',
        start_time=start_time,
        termination_time=termination_time,
    )


class TestTerminateScanjob:

    def test_unknown_scanjob(self):
        store = MagicMock(ScanJobStore)
        store.get_scanjob_by_id.side_effect = LookupError
        with pytest.raises(UnknownScanjobError):
            terminate_scanjob(store, 'unknown', 'The Message')

    def test_not_started(self):
        store = MagicMock(ScanJobStore)
        store.get_scanjob_by_id.return_value = make_scanjob(start_time=None)
        with pytest.raises(TerminateScanJobError):
            terminate_scanjob(store, 'scjb000', 'The Message')

    def test_already_terminated(self):
        store = MagicMock(ScanJobStore)
        store.get_scanjob_by_id.return_value = make_scanjob(
            start_time=datetime(
                1985, 10, 26, 1, 20, tzinfo=utc
            ),
            termination_time=datetime(
                1985, 10, 26, 1, 21, tzinfo=utc
            )
        )
        with pytest.raises(TerminateScanJobError):
            terminate_scanjob(store, 'scjb000', 'The Message')

    def test_already_ended(self):
        store = MagicMock(ScanJobStore)
        store.get_scanjob_by_id.return_value = make_scanjob(
            start_time=datetime(
                1985, 10, 26, 1, 20, tzinfo=utc
            ),
            termination_time=None,
        )
        with pytest.raises(TerminateScanJobError):
            terminate_scanjob(store, 'scjb000', 'The Message')

    def test_running_scanjob(self):
        store = MagicMock(ScanJobStore)
        store.get_scanjob_by_id.return_value = make_scanjob(
            start_time=datetime(
                1985, 10, 26, 1, 20, tzinfo=utc
            ),
            duration=timedelta(minutes=20),
        )
        now = datetime(1985, 10, 26, 1, 21, tzinfo=utc)
        with freeze_time(now):
            terminate_scanjob(store, 'scjb000', 'The Message')
        store.terminate_scanjob.assert_called_with(
            'scjb000', now, 'The Message'
        )
