from __future__ import absolute_import
from collections import namedtuple


ScannerBase = namedtuple(
    'ScannerBase',
    ['name', 'identifier', 'last_seen']
)


class Scanner(ScannerBase):
    def __new__(self, name, identifier, last_seen=None):
        return super(Scanner, self).__new__(self, name, identifier, last_seen)
