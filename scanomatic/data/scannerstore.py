from __future__ import absolute_import
from pytz import utc
import sqlalchemy as sa
from sqlalchemy.sql.expression import exists

from scanomatic.models.scanner import Scanner


class ScannerStore(object):
    class IntegrityError(Exception):
        pass

    def __init__(self, connection, dbmetadata):
        self._connection = connection
        self._table = dbmetadata.tables['scanners']

    def add(self, scanner):
        try:
            self._connection.execute(self._table.insert().values(
                name=scanner.name,
                id=scanner.identifier,
                last_seen=scanner.last_seen,
            ))
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def get_all(self):
        query = self._table.select()
        return self._get_scanners(query)

    def get_scanner_by_id(self, id_):
        query = self._table.select().where(self._table.c.id == id_)
        for scanner in self._get_scanners(query):
            return scanner
        else:
            raise KeyError(id)

    def _get_scanners(self, query):
        for row in self._connection.execute(query):
            print(row)
            if row['last_seen'] is not None:
                last_seen = row['last_seen'].astimezone(utc)
            else:
                last_seen = None
            yield Scanner(
                name=row['name'],
                identifier=row['id'],
                last_seen=last_seen,
            )

    def has_scanner_with_id(self, id_):
        query = exists(
            self._table.select().where(self._table.c.id == id_)
        ).select()
        return self._connection.execute(query).scalar()

    def update_scanner_status(self, id_, last_seen):
        result = self._connection.execute(
            self._table.update().where(self._table.c.id == id_).values(
                last_seen=last_seen,
            )
        )
        if result.rowcount == 0:
            raise KeyError(id_)
