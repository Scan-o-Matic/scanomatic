from __future__ import absolute_import
import os


def get_database_url():
    return 'postgresql://{user}@{host}:{port}/{database}'.format(
        database=os.getenv('PGDATABASE', 'scanomatic'),
        host=os.getenv('PGHOST', 'localhost'),
        port=os.getenv('PGPORT', 5432),
        user=os.getenv('PGUSER', 'scanomatic'),
    )
