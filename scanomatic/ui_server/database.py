from __future__ import absolute_import
from collections import namedtuple

from flask import g, current_app
from sqlalchemy import create_engine


DBState = namedtuple('dbstate', ['connection', 'transaction'])


def setup(app):
    app.config['dbengine'] = create_engine(app.config['DATABASE_URL'])

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
    global x
    if 'dbstate' not in g:
        connection = current_app.config['dbengine'].connect()
        g.dbstate = DBState(connection, connection.begin())
    assert isinstance(g.dbstate, DBState)
    return g.dbstate.connection
