from __future__ import absolute_import

from collections import namedtuple
import warnings

from flask import current_app, g
import sqlalchemy as sa

from scanomatic.data.calibrationstore import CalibrationStore
from scanomatic.data.scanjobstore import ScanJobStore
from scanomatic.data.scannerstore import ScannerStore
from scanomatic.data.scanstore import ScanStore
from scanomatic.data.util import get_database_metadata

DBState = namedtuple('dbstate', ['connection', 'transaction'])


def setup(app):
    app.config['dbengine'] = sa.create_engine(app.config['DATABASE_URL'])
    app.config['dbmetadata'] = get_database_metadata(app.config['dbengine'])

    @app.teardown_appcontext
    def close_database_connection(exception):
        dbstate = g.get('dbstate')
        if dbstate is None:
            return
        if exception is None:
            dbstate.transaction.commit()
        else:
            dbstate.transaction.rollback()


def connect():
    if 'dbstate' not in g:
        connection = current_app.config['dbengine'].connect()
        g.dbstate = DBState(connection, connection.begin())
    assert isinstance(g.dbstate, DBState)
    return g.dbstate.connection


def getscannerstore():
    return ScannerStore(connect(), current_app.config['dbmetadata'])


def getscanstore():
    return ScanStore(connect(), current_app.config['dbmetadata'])


def getscanjobstore():
    return ScanJobStore(connect(), current_app.config['dbmetadata'])


def getcalibrationstore():
    return CalibrationStore(connect(), current_app.config['dbmetadata'])
