from __future__ import absolute_import
import os

import alembic
from alembic.config import Config as AlembicConfig
import sqlalchemy as sa
import pytest

import scanomatic.data


CONFIG_PATH = os.path.join(
    os.path.dirname(scanomatic.data.__file__), 'migrations', 'alembic.ini',
)


@pytest.fixture
def database(postgresql):
    dbparams = postgresql.get_dsn_parameters()
    dburl = 'postgresql://{user}@{host}:{port}/{dbname}'.format(**dbparams)
    engine = sa.create_engine(dburl)
    engine.execute('CREATE EXTENSION btree_gist;')
    alembic_cfg = AlembicConfig(CONFIG_PATH)
    alembic_cfg.set_main_option('sqlalchemy.url', dburl)
    alembic.command.upgrade(alembic_cfg, "head")
    return dburl
