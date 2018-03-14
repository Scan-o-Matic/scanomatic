from __future__ import absolute_import

from contextlib import contextmanager
import os
import warnings

import sqlalchemy as sa


def get_database_url():
    return 'postgresql://{user}@{host}:{port}/{database}'.format(
        database=os.getenv('PGDATABASE', 'scanomatic'),
        host=os.getenv('PGHOST', 'localhost'),
        port=os.getenv('PGPORT', 5432),
        user=os.getenv('PGUSER', 'scanomatic'),
    )


def get_database_metadata(dbengine):
    dbmetadata = sa.MetaData(bind=dbengine)
    with warnings.catch_warnings():
        # Avoid spurious warnings: sqlalchemy can't reflect some constraints
        # from the database and emits a warning about this.
        # This is fine as long as we don't try to recreate the tables
        # based on the reconstructed schema.
        warnings.simplefilter("ignore", category=sa.exc.SAWarning)
        dbmetadata.reflect()
    return dbmetadata


@contextmanager
def store_from_env(store_class):
    dbengine = sa.create_engine(get_database_url())
    dbmetadata = get_database_metadata(dbengine)
    connection = dbengine.connect()
    transaction = connection.begin()
    try:
        yield store_class(connection, dbmetadata)
    except:
        transaction.rollback()
        raise
    else:
        transaction.commit()
    finally:
        connection.close()
