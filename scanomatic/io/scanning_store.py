from __future__ import absolute_import
from datetime import datetime
from collections import namedtuple
import os
import pytz


class ScanJobCollisionError(ValueError):
    pass


class ScanJobUnknownError(ValueError):
    pass


class DuplicateIdError(ValueError):
    pass


class UnknownIdError(ValueError):
    pass


Scanner = namedtuple(
    'Scanner',
    ['name', 'identifier']
)
ScannerStatus = namedtuple(
    'ScannerStatus',
    ['job', 'server_time']
)


class ScanningStore:
    def __init__(self):
        if not int(os.environ.get('SOM_HIDE_TEST_SCANNERS', '0')):
            self._scanners = {
                '9a8486a6f9cb11e7ac660050b68338ac': Scanner(
                    'Scanner one',
                    '9a8486a6f9cb11e7ac660050b68338ac',
                ),
                '350986224086888954': Scanner(
                    'Scanner two',
                    '350986224086888954',
                ),
            }
        else:
            self._scanners = {}
        self._scanner_statuses = {scanner: [] for scanner in self._scanners}
        self._scanjobs = {}
        self._scans = {}

    def has_scanner(self, identifier):
        return identifier in self._scanners

    def get_scanner(self, identifier):
        return self._scanners[identifier]

    def get_scanner_by_name(self, name):
        scanners = [
            scanner for scanner in self._scanners
            if self._scanners[scanner].name == name
        ]
        if len(scanners) > 1:
            raise ValueError("Duplicate name '{}' in scanner list".format(
                name
            ))
        elif len(scanners) == 0:
            return None
        else:
            return scanners[0]

    def add_scanner(self, scanner):
        if self.has_scanner(scanner.identifier):
            raise ValueError(
                "Cannot add duplicate scanner with id '{}'".format(
                    scanner.identifier)
            )
        elif self.get_scanner_by_name(scanner.name) is not None:
            raise ValueError(
                "Cannot add duplicate scanner with name '{}'".format(
                    scanner.name)
            )
        self._scanners[scanner.identifier] = scanner
        self._scanner_statuses[scanner.identifier] = []

    def get_free_scanners(self):
        return [
            scanner for scanner in self._scanners.values()
            if self.has_current_scanjob(
                scanner.identifier,
                datetime.now(pytz.utc)
            ) is False
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
