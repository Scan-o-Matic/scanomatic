"""create scanjob table

Revision ID: 39da65401ca3
Revises: 6736f58587af
Create Date: 2018-02-26 11:29:24.432510

"""
from __future__ import absolute_import
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.sql import quoted_name


# revision identifiers, used by Alembic.
revision = '39da65401ca3'
down_revision = '6736f58587af'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'scanjobs',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('name', sa.Text(), unique=True, nullable=False),
        sa.Column('duration', sa.Interval(), nullable=False),
        sa.Column('interval', sa.Interval(), nullable=False),
        sa.Column(
            'scanner_id',
            sa.Text(),
            sa.ForeignKey('scanners.id', name='fk_scanjob_scanner_id'),
            nullable=False,
        ),
        sa.Column('start_time', sa.DateTime(timezone=True)),

        # Create a constraint that prevents scanjobs to overlap for the same
        # scanner id.  This a PostgreSQL specific construct.  quoted_name is
        # used to pass a raw expression because it seems that the tsrange
        # function is not available in sqlalchemy.
        ExcludeConstraint(
            ('scanner_id', '='),
            (sa.Column(quoted_name(
                '''
                    tsrange(
                        start_time AT TIME ZONE 'UTC',
                        start_time AT TIME ZONE 'UTC' + duration,
                        '[]'
                    )
                ''', quote=False)), '&&'),
            where=(sa.Column('start_time').isnot(None)),
            name='exclude_overlapping_scanjobs',
        ),
    )


def downgrade():
    pass
