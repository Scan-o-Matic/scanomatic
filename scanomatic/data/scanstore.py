from __future__ import absolute_import
from pytz import utc
import sqlalchemy as sa

from scanomatic.models.scan import Scan


class ScanStore(object):
    class IntegrityError(Exception):
        pass

    def __init__(self, connection, metadata):
        self._connection = connection
        self._table = metadata.tables['scans']

    def add_scan(self, scan):
        try:
            self._connection.execute(
                self._table.insert().values(
                    id=scan.identifier,
                    start_time=scan.start_time,
                    end_time=scan.end_time,
                    digest=scan.digest,
                    scanjob_id=scan.scanjob_id,
                )
            )
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def get_scan_by_id(self, id_):
        query = self._table.select().where(self._table.c.id == id_)
        for scan in self._get_scans(query):
            return scan
        else:
            raise KeyError(id_)

    def get_all_scans(self):
        query = self._table.select()
        return self._get_scans(query)

    def _get_scans(self, query):
        for row in self._connection.execute(query):
            yield Scan(
                id=row['id'],
                start_time=row['start_time'].astimezone(utc),
                end_time=row['end_time'].astimezone(utc),
                digest=row['digest'],
                scanjob_id=row['scanjob_id'],
            )
