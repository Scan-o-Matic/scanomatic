from __future__ import absolute_import
from collections import namedtuple


class ScanJobCollisionError(ValueError):
    pass


class ScanJobUnknownError(ValueError):
    pass


ScanJob = namedtuple(
    'ScanJob',
    ['identifier', 'name', 'duration', 'interval', 'scanner']
)


class ScanningStore:
    def __init__(self):
        self._scanners = {
            'Test': {
                'name': 'Test',
                'power': False,
                'owner': None,
            },
        }
        self._scanjobs = {}

    def has_scanner(self, name):
        return name in self._scanners

    def get_scanner(self, name):
        return self._scanners[name]

    def get_free_scanners(self):
        return [
            scanner for scanner in self._scanners.values()
            if scanner['owner'] is None
        ]

    def get_all_scanners(self):
        return list(self._scanners.values())

    def add_scanjob(self, job):
        if job.identifier in self._scanjobs:
            raise ScanJobCollisionError(
                "{} already used".format(job.identifier)
            )
        self._scanjobs[job.identifier] = job

    def remove_scanjob(self, identifier):
        if identifier in self._scanjobs:
            del self._scanjobs[identifier]
        else:
            raise ScanJobUnknownError(
                "{} is not a known job".format(identifier)
            )

    def get_scanjobs(self):
        return list(self._scanjobs.values())

    def get_scanjob_ids(self):
        return list(self._scanjobs.keys())

    def exists_scanjob_with(self, key, value):
        for job in self._scanjobs.values():
            if getattr(job, key) == value:
                return True
        return False
