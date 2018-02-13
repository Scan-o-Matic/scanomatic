from __future__ import absolute_import
from collections import namedtuple
from datetime import datetime

from prometheus_client import Gauge, Counter
import pytz

from scanomatic.io.scanning_store import DuplicateNameError
from scanomatic.models.scanner import Scanner
from scanomatic.models.scannerstatus import ScannerStatus
from scanomatic.util.datetime import timestamp
from scanomatic.util.generic_name import get_generic_name


SCANNER_CURRENT_JOBS = Gauge(
    'scanner_current_jobs',
    'Number of job currently executed',
    ['scanner']
)
SCANNER_QUEUED_UPLOADS = Gauge(
    'scanner_queued_uploads',
    'Number of images queued for upload',
    ['scanner']
)
SCANNER_START_TIME = Gauge(
    'scanner_start_time_seconds',
    'Start time of the daemon since unix epoch in seconds',
    ['scanner']
)
SCANNER_LAST_STATUS_UPDATE_TIME = Gauge(
    'scanner_last_status_update_time_seconds',
    'Number of seconds since unix epoch for the last status update',
    ['scanner']
)
SCANNER_STATUS_UPDATES = Counter(
    'scanner_status_updates_total',
    'Number of status updates',
    ['scanner']
)


def UpdateScannerStatusError(Exception):
    pass


UpdateScannerStatusResult = namedtuple('UpdateScannerStatusResult', [
    'new_scanner'
])


def update_scanner_status(
    db, scanner_id, job, start_time, next_scheduled_scan, images_to_send
):
    if not db.has_scanner(scanner_id):
        _add_scanner(db, scanner_id)
        new_scanner = True
    else:
        new_scanner = False
    status = ScannerStatus(
        server_time=datetime.now(pytz.utc),
        job=job,
        start_time=start_time,
        next_scheduled_scan=next_scheduled_scan,
        images_to_send=images_to_send,
    )
    db.add_scanner_status(scanner_id, status)
    SCANNER_CURRENT_JOBS.labels(scanner=scanner_id).set(job is not None)
    SCANNER_QUEUED_UPLOADS.labels(scanner=scanner_id).set(images_to_send)
    SCANNER_START_TIME.labels(scanner=scanner_id).set(timestamp(start_time))
    SCANNER_LAST_STATUS_UPDATE_TIME.labels(scanner=scanner_id).set_to_current_time()
    SCANNER_STATUS_UPDATES.labels(scanner=scanner_id).inc()
    return UpdateScannerStatusResult(new_scanner=new_scanner)


def _add_scanner(db, scanner_id):
    try:
        name = get_generic_name()
        db.add_scanner(Scanner(name, scanner_id))
    except DuplicateNameError:
        UpdateScannerStatusError(
            "Failed to create scanner, please try again",
        )
