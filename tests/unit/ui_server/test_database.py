from __future__ import absolute_import
from flask import Flask
from mock import patch, MagicMock
import pytest

from scanomatic.ui_server import database


@pytest.fixture
def engine():
    return MagicMock()


@pytest.fixture
def app(engine):
    app = Flask('mytestapp')
    app.config['DATABASE_URL'] = 'mydb://...'
    with patch(
        'scanomatic.ui_server.database.create_engine',
        return_value=engine,
    ):
        database.setup(app)
    return app


class TestConnect(object):
    def test_return_transaction(self, app, engine):
        with app.app_context():
            conn = database.connect()
            assert conn is engine.connect()

    def test_cache_connection(self, app, engine):
        with app.app_context():
            engine.connect.reset_mock()
            conn1 = database.connect()
            conn2 = database.connect()
            assert conn1 is conn2
            engine.connect.assert_called_once()


class TestTeardown(object):
    def test_commit_on_success(self, app, engine):
        with app.app_context():
            database.connect()
        engine.connect().begin().commit.assert_called()
        engine.connect().begin().rollback.assert_not_called()

    def test_rollback_on_exception(self, app, engine):
        class MyError(Exception):
            pass

        try:
            with app.app_context():
                database.connect()
                raise MyError
        except MyError:
            pass
        engine.connect().begin().commit.assert_not_called()
        engine.connect().begin().rollback.assert_called()

    def test_do_nothing_if_no_connection(self, app, engine):
        with app.app_context():
            pass
        engine.connect().begin().commit.assert_not_called()
        engine.connect().begin().rollback.assert_not_called()
