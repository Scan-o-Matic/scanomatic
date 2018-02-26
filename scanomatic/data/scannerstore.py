from __future__ import absolute_import
import sqlalchemy as sa
from sqlalchemy.sql.expression import exists

from .tables import scanners
from scanomatic.models.scanner import Scanner


class ScannerStore(object):
    class IntegrityError(Exception):
        pass

    def __init__(self, connection):
        self._connection = connection

    def add(self, scanner):
        try:
            self._connection.execute(scanners.insert().values(
                name=scanner.name, id=scanner.identifier,
            ))
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def get_all(self):
        query = scanners.select()
        for row in self._connection.execute(query):
            yield Scanner(name=row['name'], identifier=row['id'])

    def get_scanner_by_id(self, id_):
        query = scanners.select().where(scanners.c.id == id_)
        for row in self._connection.execute(query):
            return Scanner(name=row['name'], identifier=row['id'])
        else:
            raise KeyError(id)

    def has_scanner_with_id(self, id_):
        query = exists(scanners.select().where(scanners.c.id == id_)).select()
        return self._connection.execute(query).scalar()
