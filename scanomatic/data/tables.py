from __future__ import absolute_import
from sqlalchemy import (
    Table, MetaData, Column, ForeignKey,
    DateTime, Interval, Text,
)
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.sql import quoted_name


metadata = MetaData()


scanners = Table(
    'scanners', metadata,
    Column('id', Text(), primary_key=True),
    Column('name', Text(), unique=True),
)

scanjobs = Table(
    'scanjobs', metadata,
    Column('id', Text(), primary_key=True),
    Column('name', Text(), unique=True, nullable=False),
    Column('duration', Interval(), nullable=False),
    Column('interval', Interval(), nullable=False),
    Column('scanner_id', Text(), ForeignKey('scanners.id'), nullable=False),
    Column('start_time', DateTime(timezone=True)),

    # Create a constraint that prevents scanjobs to overlap for the same
    # scanner id.  This a PostgreSQL specific construct.  quoted_name is used
    # to pass a raw expression because it seems that the tsrange function is
    # not available in sqlalchemy.
    ExcludeConstraint(
        ('scanner_id', '='),
        (Column(quoted_name(
            '''
                tsrange(
                    start_time AT TIME ZONE 'UTC',
                    start_time AT TIME ZONE 'UTC' + duration,
                    '[]'
                )
            ''', quote=False)), '&&'),
        where=(Column('start_time').isnot(None)),
        name='exclude_overlapping_scanjobs',
    ),
)
