from __future__ import absolute_import
import sqlalchemy as sa
from sqlalchemy.sql.expression import exists

from scanomatic.models.scanner import Scanner


class ScannerStore(object):
    class IntegrityError(Exception):
        pass

    def __init__(self, connection, metadata):
        self._connection = connection
        self._table = metadata.tables['scanners']

    def add(self, scanner):
        try:
            self._connection.execute(self._table.insert().values(
                name=scanner.name, id=scanner.identifier,
            ))
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def get_all(self):
        query = self._table.select()
        for row in self._connection.execute(query):
            yield Scanner(name=row['name'], identifier=row['id'])

    def get_scanner_by_id(self, id_):
        query = self._table.select().where(self._table.c.id == id_)
        for row in self._connection.execute(query):
            return Scanner(name=row['name'], identifier=row['id'])
        else:
            raise KeyError(id)

    def has_scanner_with_id(self, id_):
        query = exists(
            self._table.select().where(self._table.c.id == id_)
        ).select()
        return self._connection.execute(query).scalar()
