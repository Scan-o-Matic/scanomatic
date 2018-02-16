from collections import namedtuple

ScanBase = namedtuple('ScanBase', [
    'id', 'scanjob_id', 'start_time', 'end_time', 'digest',
])


class Scan(ScanBase):
    @property
    def identifier(self):
        return self.id
