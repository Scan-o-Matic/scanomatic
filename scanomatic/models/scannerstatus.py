from __future__ import absolute_import
from collections import namedtuple


ScannerStatus = namedtuple('ScannerStatus', [
    'job',
    'server_time',
    'start_time',
    'images_to_send',
    'next_scheduled_scan'
])
