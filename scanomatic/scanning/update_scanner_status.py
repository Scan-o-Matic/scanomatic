from __future__ import absolute_import
from collections import namedtuple
from datetime import datetime

import pytz

from scanomatic.io.scanning_store import DuplicateNameError
from scanomatic.models.scanner import Scanner
from scanomatic.models.scannerstatus import ScannerStatus
from scanomatic.util.generic_name import get_generic_name


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
    return UpdateScannerStatusResult(new_scanner=new_scanner)


def _add_scanner(db, scanner_id):
    try:
        name = get_generic_name()
        db.add_scanner(Scanner(name, scanner_id))
    except DuplicateNameError:
        UpdateScannerStatusError(
            "Failed to create scanner, please try again",
        )
