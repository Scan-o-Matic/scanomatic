from datetime import datetime, timedelta

from mock import MagicMock
import pytest
from pytz import utc

from scanomatic.data.scanjobstore import ScanJobStore
from scanomatic.models.scanjob import ScanJob
from scanomatic.scanning.delete_scanjob import (
    DeleteScanjobError, delete_scanjob
)
from scanomatic.scanning.exceptions import UnknownScanjobError


def make_scanjob(start_time=None):
    return ScanJob(
        identifier='scjb001',
        name='Test Scanjob',
        duration=timedelta(minutes=20),
        interval=timedelta(minutes=5),
        scanner_id='scnr002',
        start_time=start_time,
    )


def test_delete_unstarted_scanjob():
    scanjob_store = MagicMock(spec=ScanJobStore)
    scanjob_store.get_scanjob_by_id.return_value = make_scanjob(
        start_time=None,
    )
    delete_scanjob(scanjob_store, 'scjb001')
    scanjob_store.delete_scanjob.assert_called_with('scjb001')


def test_delete_unknown_scanjob():
    scanjob_store = MagicMock(spec=ScanJobStore)
    scanjob_store.get_scanjob_by_id.side_effect = LookupError
    with pytest.raises(
        UnknownScanjobError, match='No scanjob with id "scjb001"'
    ):
        delete_scanjob(scanjob_store, 'scjb001')
    scanjob_store.delete_scanjob.assert_not_called()


def test_delete_started_scanjob():
    scanjob_store = MagicMock(spec=ScanJobStore)
    scanjob_store.get_scanjob_by_id.return_value = make_scanjob(
        start_time=datetime(
            1985, 10, 26, 1, 20, tzinfo=utc
        )
    )
    with pytest.raises(
        DeleteScanjobError,
        match='Scanjob scjb001 has been started, cannot delete'
    ):
        delete_scanjob(scanjob_store, 'scjb001')
    scanjob_store.delete_scanjob.assert_not_called()
