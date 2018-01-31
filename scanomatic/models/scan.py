from collections import namedtuple

Scan = namedtuple('Scan', [
    'id', 'scanjob_id', 'start_time', 'end_time', 'digest',
])
