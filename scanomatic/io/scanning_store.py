from __future__ import absolute_import
from collections import defaultdict
from datetime import datetime
import pytz

from scanomatic.models.scan import Scan
from scanomatic.models.scanjob import ScanJob
from scanomatic.models.scanner import Scanner
from scanomatic.models.scannerstatus import ScannerStatus


class DuplicateIdError(ValueError):
    pass


class DuplicateNameError(ValueError):
    pass


class UnknownIdError(ValueError):
    def __init__(self, klass, identifier):
        super(UnknownIdError, self).__init__()
        self._klass = klass
        self._identifier = identifier

    def __str__(self):
        return 'Unknown {} id: {}'.format(
            self._klass.__name__, self._identifier
        )


class ScanningStore:
    def __init__(self):
        self._stores = {
            Scan: {},
            ScanJob: {},
            Scanner: {},
            ScannerStatus: defaultdict(list),
        }

    def _get_store(self, klass):
        try:
            return self._stores[klass]
        except KeyError:
            raise TypeError('No store for object type {}'.format(klass))

    def add(self, item):
        store = self._get_store(type(item))
        if item.identifier in store:
            raise DuplicateIdError(type(item), item.identifier)
        if type(item) is Scanner and self.exists(Scanner, name=item.name):
            raise DuplicateNameError(
                "Cannot add duplicate scanner with name '{}'".format(
                    item.name)
            )
        if type(item) is Scan and not self.exists(ScanJob, item.scanjob_id):
            raise UnknownIdError(ScanJob, item.scanjob_id)
        self._get_store(type(item))[item.identifier] = item

    def get(self, klass, id_):
        try:
            return self._get_store(klass)[id_]
        except KeyError:
            raise UnknownIdError(klass, id_)

    def find(self, klass, **constraints):
        store = self._get_store(klass)
        for item in store.values():
            for key in constraints:
                if getattr(item, key) != constraints[key]:
                    break
            else:
                yield item

    def exists(self, klass, identifier=None, **constraints):
        if identifier is not None:
            return identifier in self._get_store(klass)
        for item in self.find(klass, **constraints):
            return True
        return False

    def update(self, item):
        if not self.exists(type(item), item.identifier):
            raise UnknownIdError(type(item), item.identifier)
        self._get_store(type(item))[item.identifier] = item

    def get_free_scanners(self):
        for scanner in self.find(Scanner):
            if not self.has_current_scanjob(
                scanner.identifier,
                datetime.now(pytz.utc)
            ):
                yield scanner

    def get_current_scanjob(self, scanner_id, timepoint):
        for job in self.find(ScanJob, scanner_id=scanner_id):
            if job.is_active(timepoint):
                return job

    def has_current_scanjob(self, scanner_id, timepoint):
        return self.get_current_scanjob(scanner_id, timepoint) is not None

    def get_scanner_status_list(self, scanner_id):
        if self.exists(Scanner, scanner_id):
            return self._get_store(ScannerStatus)[scanner_id]
        else:
            raise UnknownIdError(Scanner, scanner_id)

    def get_latest_scanner_status(self, scanner_id):
        try:
            return self.get_scanner_status_list(scanner_id)[-1]
        except IndexError:
            return None

    def add_scanner_status(self, scanner_id, status):
        self.get_scanner_status_list(scanner_id).append(status)
