from __future__ import absolute_import
from collections import namedtuple


class ScanJobCollisionError(ValueError):
    pass


class ScanJobUnknownError(ValueError):
    pass


Scanner = namedtuple(
    'Scanner',
    ['name', 'power', 'owner', 'identifier']
)


class ScanningStore:
    def __init__(self):
        self._scanners = {
            '9a8486a6f9cb11e7ac660050b68338ac': Scanner(
                'Test',
                False,
                None,
                '9a8486a6f9cb11e7ac660050b68338ac'
            ),
        }
        self._scanjobs = {}

    def has_scanner(self, identifier):
        return identifier in self._scanners

    def get_scanner(self, identifier):
        return self._scanners[identifier]

    def get_free_scanners(self):
        return [
            scanner for scanner in self._scanners.values()
            if scanner.owner is None
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

    def get_scanjob(self, identifier):
        try:
            return self._scanjobs[identifier]
        except KeyError:
            raise ScanJobUnknownError(identifier)

    def update_scanjob(self, job):
        if job.identifier not in self._scanjobs:
            raise ScanJobUnknownError(job.identifier)
        self._scanjobs[job.identifier] = job

    def get_all_scanjobs(self):
        return list(self._scanjobs.values())

    def get_scanjob_ids(self):
        return list(self._scanjobs.keys())

    def exists_scanjob_with(self, key, value):
        for job in self._scanjobs.values():
            if getattr(job, key) == value:
                return True
        return False

    def get_current_scanjob(self, scanner_id, timepoint):
        for job in self._scanjobs.values():
            if job.scanner_id == scanner_id and job.is_active(timepoint):
                return job

    def has_current_scanjob(self, scanner_id, timepoint):
        return self.get_current_scanjob(scanner_id, timepoint) is not None
