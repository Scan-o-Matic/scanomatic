from __future__ import absolute_import
from collections import defaultdict

from scanomatic.models.scan import Scan
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
            ScannerStatus: defaultdict(list),
        }

    def _get_store(self, klass):
        try:
            return self._stores[klass]
        except KeyError:
            raise TypeError('No store for object type {}'.format(klass))

    def add(self, item):
        klass = type(item)
        store = self._get_store(klass)
        if item.identifier in store:
            raise DuplicateIdError(klass, item.identifier)
        store[item.identifier] = item

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
            constraints['identifier'] = identifier
        for item in self.find(klass, **constraints):
            return True
        return False

    def update(self, item):
        if not self.exists(type(item), item.identifier):
            raise UnknownIdError(type(item), item.identifier)
        self._get_store(type(item))[item.identifier] = item

    def get_scanner_status_list(self, scanner_id):
        return self._get_store(ScannerStatus)[scanner_id]

    def get_latest_scanner_status(self, scanner_id):
        try:
            return self.get_scanner_status_list(scanner_id)[-1]
        except IndexError:
            return None

    def add_scanner_status(self, scanner_id, status):
        self.get_scanner_status_list(scanner_id).append(status)
