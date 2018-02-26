from pytz import utc
import sqlalchemy as sa
from sqlalchemy.sql import and_
from sqlalchemy.sql.expression import exists

from scanomatic.models.scanjob import ScanJob


class ScanJobStore(object):
    class IntegrityError(Exception):
        pass

    def __init__(self, connection, metadata):
        self._connection = connection
        self._table = metadata.tables['scanjobs']

    def add_scanjob(self, scanjob):
        try:
            self._connection.execute(self._table.insert().values(
                duration=scanjob.duration,
                id=scanjob.identifier,
                interval=scanjob.interval,
                name=scanjob.name,
                scanner_id=scanjob.scanner_id,
                start_time=scanjob.start_time,
            ))
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def set_scanjob_start_time(self, id_, start_time):
        try:
            self._connection.execute(
                self._table.update()
                .where(self._table.c.id == id_)
                .values(start_time=start_time)
            )
        except sa.exc.IntegrityError as e:
            raise self.IntegrityError(e)

    def get_scanjob_by_id(self, id_):
        query = self._table.select().where(self._table.c.id == id_)
        for scanjob in self._get_scanjobs(query):
            return scanjob
        else:
            raise KeyError(id_)

    def get_all_scanjobs(self):
        query = self._table.select()
        return self._get_scanjobs(query)

    def _get_scanjobs(self, query):
        for row in self._connection.execute(query):
            if row['start_time'] is not None:
                start_time = row['start_time'].astimezone(utc)
            else:
                start_time = None
            yield ScanJob(
                name=row['name'],
                identifier=row['id'],
                duration=row['duration'],
                interval=row['interval'],
                scanner_id=row['scanner_id'],
                start_time=start_time,
            )

    def has_scanjob_with_name(self, name):
        query = (
            exists(self._table.select().where(self._table.c.name == name))
            .select()
        )
        return self._connection.execute(query).scalar()

    def get_current_scanjob_for_scanner(self, scanner_id, when):
        query = self._table.select().where(
            and_(
                self._table.c.scanner_id == scanner_id,
                self._table.c.start_time <= when,
                self._table.c.start_time + self._table.c.duration >= when,
            )
        )
        for scanjob in self._get_scanjobs(query):
            return scanjob
